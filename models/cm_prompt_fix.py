# src/models/cm_prompt_fix.py

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from src.configs.cm_prompt import PromptSpec, compute_prompt_positions, target_mask_position_from_end


class CMPromptFIX(RobertaForMaskedLM):

    def __init__(
        self,
        config,
        u_dim: int = 1024,
        l_dim: int = 9,
        f_dim: int = 1024,
        ple: int = 3,
        plp: int = 3,
        pre: int = 3,
        prp: int = 3,
        dropout: float = 0.1,
        emotion_token_ids: Optional[List[int]] = None,
    ):
        super().__init__(config)

        self.spec = PromptSpec(ple=ple, plp=plp, pre=pre, prp=prp)
        self.dim = 512

        self.b_emb = self.get_input_embeddings()
        self.emb = nn.Embedding(512, self.dim)

        # commonsense encoder
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8, batch_first=True)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)

        # semantic encoder
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8, batch_first=True)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)

        # blend two features
        self.bi_lstm = nn.LSTM(2 * self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        # global
        self.hidden_size = 512
        self.lstm_store = nn.LSTMCell(self.dim, self.hidden_size)
        self.lstm_affect = nn.LSTMCell(self.dim, self.hidden_size)

        # common ground
        self.cdim = 1792
        self.common_hidden = 1792
        self.cg_store = nn.LSTMCell(self.cdim, self.common_hidden)
        self.cg_affect = nn.LSTMCell(self.cdim, self.common_hidden)

        self.line_cr = nn.Linear(512, 512)
        self.line_crp = nn.Linear(1024, self.dim)  # save
        self.line1 = nn.Linear(1024, 1024)         # save
        self.line2 = nn.Linear(1024, 1024)         # save
        self.line3 = nn.Linear(1024, 1024)         # save

        self.lineu = nn.Linear(u_dim, 512)
        self.lineu2 = nn.Linear(u_dim, 256)
        self.linel = nn.Linear(l_dim, 256)         # save
        self.linef = nn.Linear(f_dim, 256)
        self.linee = nn.Linear(1024, self.dim * (ple + pre))  # save
        self.linep = nn.Linear(1024, self.dim * (plp + prp))  # save
        self.global_c = nn.Linear(512, 1024)
        self.common_g = nn.Linear(1792, 1024)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        # 12  prompt token  delta
        self.linenp = nn.Linear(1024, self.dim * (ple + pre + plp + prp))


        self.emotion_token_ids = emotion_token_ids if emotion_token_ids is not None else [7974, 2755, 2490, 17437, 5823, 30883, 6378]

    def _build_position_embeddings(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:

        device = input_ids.device
        B, L = input_ids.shape
        spec = self.spec

        # left prompt embeddings: indices 1..ple, ple+1..ple+plp
        ple_idx = torch.arange(1, spec.ple + 1, device=device).unsqueeze(0).expand(B, -1)
        plp_idx = torch.arange(spec.ple + 1, spec.ple + spec.plp + 1, device=device).unsqueeze(0).expand(B, -1)

        # right prompt indices: based on absolute position near end
        prp_start = L - spec.pre - spec.prp - 1
        prp_end = L - spec.pre - 1
        pre_start = L - spec.pre - 1
        pre_end = L - 1

        prp_idx = torch.arange(prp_start, prp_end, device=device).unsqueeze(0).expand(B, -1)
        pre_idx = torch.arange(pre_start, pre_end, device=device).unsqueeze(0).expand(B, -1)

        ple = self.emb(ple_idx)  # (B, ple, 512)
        plp = self.emb(plp_idx)  # (B, plp, 512)
        prp = self.emb(prp_idx)  # (B, prp, 512)
        pre = self.emb(pre_idx)  # (B, pre, 512)

        pos = compute_prompt_positions(L, spec)  # length = ple+plp+pre+prp
        return ple, plp, prp, pre, pos

    def forward(
        self,
        X: torch.Tensor,                # (1, T, 1024)
        C: List[torch.Tensor],          # list length loc+1, each (1792,)
        r: torch.Tensor,                # (512,) or (1,512) float
        cur_speakers: List[str],        # role list
        lf: torch.Tensor,               # (1, T, 1024) tabtext
        loc: int,                       # utterance index i
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        device = input_ids.device
        B, L = input_ids.shape
        assert B == 1, 

        ple, plp, prp, pre, pos = self._build_position_embeddings(input_ids)

        # utterance -> U
        U_512 = self.lineu(X.to(device))  # (1, T, 512)

        # ---------- global ----------
        h_o = torch.zeros(1, self.hidden_size, device=device)
        c_o = torch.zeros(1, self.hidden_size, device=device)
        h_n, c_n = h_o, c_o

        speaker_id = loc + 1
        now_speaker = cur_speakers[speaker_id]
        for i in range(speaker_id):
            h_o, c_o = h_n, c_n
            u_i = U_512[0, i].unsqueeze(0)  # (1,512)
            h_store, c_store = self.lstm_store(u_i, (h_o, c_o))
            h_aff, c_aff = self.lstm_affect(u_i, (h_o, c_o))
            m = 1.0 if (cur_speakers[i] == now_speaker) else 0.0
            h_n = m * h_store + (1.0 - m) * h_aff
            c_n = m * c_store + (1.0 - m) * c_aff

        global_f = self.global_c(h_n)  # (1,1024)

        # ---------- common ground ----------
        h_o = torch.zeros(1, self.common_hidden, device=device)
        c_o = torch.zeros(1, self.common_hidden, device=device)
        h_n, c_n = h_o, c_o
        for i in range(speaker_id):
            h_o, c_o = h_n, c_n
            c_i = C[i].to(device).unsqueeze(0)  # (1,1792)
            h_store, c_store = self.cg_store(c_i, (h_o, c_o))
            h_aff, c_aff = self.cg_affect(c_i, (h_o, c_o))
            m = 1.0 if (cur_speakers[i] == now_speaker) else 0.0
            h_n = m * h_store + (1.0 - m) * h_aff
            c_n = m * c_store + (1.0 - m) * c_aff

        common_g = self.common_g(h_n)  # (1,1024)

        # ---------- personal feature ----------
        LF = self.linef(lf.to(device))  # (1, T, 256)

        # ---------- semantic prompt generation ----------
        U_256 = self.lineu2(X.to(device))  # (1, T, 256)

        p_p_1 = torch.cat((plp, prp), dim=1)  # (1, plp+prp, 512)
        p_p_2 = torch.cat((ple, pre), dim=1)  # (1, ple+pre, 512)
        p_p = torch.cat((p_p_1, p_p_2), dim=1)  # (1, 12, 512)

        # cicero r -> (1,512)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        com_r = self.line_cr(r.to(device)).unsqueeze(0)  # (1,1,512)

        p_P = torch.cat((U_256, LF), dim=-1)  # (1, T, 512)
        p_P = p_P[:, loc - 1, :].unsqueeze(1)  # (1,1,512)
        p_P = torch.cat((p_P, com_r), dim=-1)  # (1,1,1024)

        # TransformerEncoder expects (B,S,E) due to batch_first=True
        p_P = self.tf_p(p_P)  # (1,1,1024)
        p_P = p_P.squeeze(1)  # (1,1024)

        P = self.linenp(p_P).reshape_as(p_p)         # (1,12,512)
        p = torch.cat((P, p_p), dim=-1)              # (1,12,1024)

        p = self.bi_lstm(p)[0]                       # (1,12,1024)

        # ---------- inject to roberta inputs_embeds ----------
        inputs_embeds = self.b_emb(input_ids)        # (1,L,1024)
        inputs_embeds[:, pos, :] = p                 # prompt injection

        inputs_embeds[:, self.spec.inject_global_idx, :] = global_f
        inputs_embeds[:, self.spec.inject_common_idx, :] = common_g

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output) 

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        pos_from_end = target_mask_position_from_end(self.spec)  # negative index
        emotion_token_logits = mlm_out.logits[:, pos_from_end, :][:, self.emotion_token_ids]  # (1,7)
        probs = self.smx(emotion_token_logits)  # (1,7)

        return probs, mlm_out.loss, mlm_out.logits
