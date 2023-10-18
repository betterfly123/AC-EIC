import torch
import transformers
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments,  BertForMaskedLM

# meld PCGEI
class CMPromptFIX(RobertaForMaskedLM):

    def __init__(self, config, u_dim=1024, l_dim=9, f_dim=1024, ple=3, plp=3, pre=3, prp=3, dropout=0.1):
        super().__init__(config)
        self.ple = ple
        self.plp = plp
        self.pre = pre
        self.prp = prp
        self.dim = 512
        self.b_emb = self.get_input_embeddings()
        self.emb = nn.Embedding(512, self.dim)
        # Commonsense features encoder
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # Semantic features encoder
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        # Model for blend the two features
        self.bi_lstm = nn.LSTM(2 * self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        # common ground
        self.cdim = 1792
        self.common_hidden = 1792
        self.cg_store = nn.LSTMCell(self.cdim, self.common_hidden)
        self.cg_affect = nn.LSTMCell(self.cdim, self.common_hidden)

        self.line_cr = nn.Linear(512, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.lineu2 = nn.Linear(u_dim, 256)
        self.linel = nn.Linear(l_dim, 256)
        self.linef = nn.Linear(f_dim, 256)
        self.linee = nn.Linear(1024, self.dim * (self.ple + self.pre))
        self.linep = nn.Linear(1024, self.dim * (self.plp + self.prp))
        self.common_g = nn.Linear(1792, 1024)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self.linenp = nn.Linear(1024, self.dim * (self.ple + self.pre + self.plp + self.prp))

    def forward(self,
                X,  # (batch, seq, dim=1024) features of utterance
                C,  # common ground
                r,  # (batch, seq, 3*768) common for listener
                cur_speakers,  # speaker name list
                lf,  # listener personal feature
                loc,  # utterance id
                input_ids=None,  # input
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,  # [CLS] -100 -100 label e(input_ids)
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # input_ids.shape[1] input length
        # print("input_ids.shape[1]:", input_ids.shape[1])
        ple = self.emb(torch.LongTensor([[i for i in range(1, self.ple + 1)]] * input_ids.shape[0]).cuda())  #(1, 3, 512)
        plp = self.emb(torch.LongTensor(
            [[i for i in range(self.ple + 1, self.ple + self.plp + 1)]] * input_ids.shape[0]).cuda())
        prp = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.pre - self.prp - 1,
                                                           input_ids.shape[1] - self.pre - 1)]] * input_ids.shape[0]
                                        ).cuda())
        pre = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.pre - 1, input_ids.shape[1] - 1)]]
                                        * input_ids.shape[0]).cuda())
        # hard prompt position
        pos = [i for i in range(1, self.ple + self.plp + 1)] + [i for i in range(input_ids.shape[1] - self.pre -
                                                                                 self.prp - 1, input_ids.shape[1] - 1)]
        # print("pos: ", pos)
        U = self.lineu(X)

        # common ground
        # print("C shape:", C[0].shape)
        h_o = torch.zeros(self.common_hidden).cuda()
        c_o = torch.zeros(self.common_hidden).cuda()
        h_n = h_o
        c_n = c_o
        for i in range(speaker_id):
            h_o = h_n
            c_o = c_n
            h_store, c_store = self.cg_store(C[i], (h_o, c_o))
            h_affect, c_affect = self.cg_affect(C[i], (h_o, c_o))
            h_n = (cur_speakers[i] == now_speakers) * h_store + (cur_speakers[i] != now_speakers) * h_affect
            c_n = (cur_speakers[i] == now_speakers) * c_store + (cur_speakers[i] != now_speakers) * c_affect
        common_g = h_n
        common_g = self.common_g(common_g)

        # personal feature
        LF = self.linef(lf) 

        # modeling [p] + Semantic
        U = self.lineu2(X)
        p_p_1 = torch.cat((plp, prp), dim=1)
        p_p_2 = torch.cat((ple, pre), dim=1)
        p_p = torch.cat((p_p_1, p_p_2), dim=1)
        com_r = self.line_cr(r) 
        com_r = torch.unsqueeze(com_r, 0)
        p_P = torch.cat((U, LF), dim=-1)
        p_P = p_P[:, loc - 1, :]  # (1, 512)
        p_P = torch.cat((p_P, com_r), dim=-1)
        p_P = self.tf_p(p_P)

        P = self.linenp(p_P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)

        # modeling for both
        p = p_p

        # soft prompt
        p = self.bi_lstm(p)[0]
        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)
        inputs_embeds[:, pos, :] = p

        # common ground
        inputs_embeds[0][7] = common_g

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
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        logit = mlm_out.logits[:, - self.ple - self.plp - 2][:, [7974, 2755, 2490, 17437, 5823, 30883, 6378]]
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits

# emorynlp PCGEI
class EmCMPromptFIX(RobertaForMaskedLM):

    def __init__(self, config, u_dim=1024, l_dim=9, f_dim=1024, ple=3, plp=3, pre=3, prp=3, dropout=0.1):
        super().__init__(config)
        self.ple = ple
        self.plp = plp
        self.pre = pre
        self.prp = prp
        self.dim = 512
        self.b_emb = self.get_input_embeddings()
        self.emb = nn.Embedding(512, self.dim)
        # Commonsense features encoder
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # Semantic features encoder
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        # Model for blend the two features
        self.bi_lstm = nn.LSTM(2 * self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        # common ground
        # self.cdim = 1792
        # self.common_hidden = 1792
        self.cdim = 3328
        self.common_hidden = 3328
        self.cg_store = nn.LSTMCell(self.cdim, self.common_hidden)
        self.cg_affect = nn.LSTMCell(self.cdim, self.common_hidden)

        self.line_cr = nn.Linear(512, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.lineu2 = nn.Linear(u_dim, 256)
        self.linel = nn.Linear(l_dim, 256)
        self.linef = nn.Linear(f_dim, 256)
        self.linee = nn.Linear(1024, self.dim * (self.ple + self.pre))
        self.linep = nn.Linear(1024, self.dim * (self.plp + self.prp))
        self.global_c = nn.Linear(512, 1024)
        self.common_g = nn.Linear(3328, 1024)
        # self.common_g = nn.Linear(1792, 1024)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self.linenp = nn.Linear(1024, self.dim * (self.ple + self.pre + self.plp + self.prp))

    def forward(self,
                X,  # (batch, seq, dim=1024) features of utterance
                C,  # common ground
                r,  # (batch, seq, 3*768) common for listener
                cur_speakers,  # speaker name list
                lf,  # listener personal feature
                loc,  # utterance id
                input_ids=None,  # input
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,  # [CLS] -100 -100 label e(input_ids)
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # input_ids.shape[1] input length
        # print("input_ids.shape[1]:", input_ids.shape[1])
        ple = self.emb(torch.LongTensor([[i for i in range(1, self.ple + 1)]] * input_ids.shape[0]).cuda())  #(1, 3, 512)
        plp = self.emb(torch.LongTensor(
            [[i for i in range(self.ple + 1, self.ple + self.plp + 1)]] * input_ids.shape[0]).cuda())
        prp = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.pre - self.prp - 1,
                                                           input_ids.shape[1] - self.pre - 1)]] * input_ids.shape[0]
                                        ).cuda())
        pre = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.pre - 1, input_ids.shape[1] - 1)]]
                                        * input_ids.shape[0]).cuda())
        # hard prompt position
        pos = [i for i in range(1, self.ple + self.plp + 1)] + [i for i in range(input_ids.shape[1] - self.pre -
                                                                                 self.prp - 1, input_ids.shape[1] - 1)]
        # print("pos: ", pos)
        U = self.lineu(X)

        # common ground
        # print("C shape:", C[0].shape)
        h_o = torch.zeros(self.common_hidden).cuda()
        c_o = torch.zeros(self.common_hidden).cuda()
        h_n = h_o
        c_n = c_o
        for i in range(speaker_id):
            h_o = h_n
            c_o = c_n
            h_store, c_store = self.cg_store(C[i], (h_o, c_o))
            h_affect, c_affect = self.cg_affect(C[i], (h_o, c_o))
            h_n = (cur_speakers[i] == now_speakers) * h_store + (cur_speakers[i] != now_speakers) * h_affect
            c_n = (cur_speakers[i] == now_speakers) * c_store + (cur_speakers[i] != now_speakers) * c_affect
        common_g = h_n
        common_g = self.common_g(common_g) 

        # personal feature
        LF = self.linef(lf) 

        # modeling [p] + Semantic
        U = self.lineu2(X)  
        p_p_1 = torch.cat((plp, prp), dim=1)
        p_p_2 = torch.cat((ple, pre), dim=1)
        p_p = torch.cat((p_p_1, p_p_2), dim=1) 
        com_r = self.line_cr(r) 
        com_r = torch.unsqueeze(com_r, 0)
        p_P = torch.cat((U, LF), dim=-1) 
        p_P = p_P[:, loc - 1, :]  # (1, 512)
        p_P = torch.cat((p_P, com_r), dim=-1) 
        p_P = self.tf_p(p_P)

        P = self.linenp(p_P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)

        # modeling for both
        p = p_p

        # soft prompt
        p = self.bi_lstm(p)[0]
        inputs_embeds = self.b_emb(input_ids)
        inputs_embeds[:, pos, :] = p

        inputs_embeds[0][7] = common_g

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
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        logit = mlm_out.logits[:, - self.ple - self.plp - 2][:, [5823, 7758, 7053, 7974, 17437, 2247, 2490]]
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits

#  CPED  PCGEI
class CpedCMPromptFIX(RobertaForMaskedLM):

    def __init__(self, config, u_dim=1024, l_dim=9, f_dim=1024, ple=3, plp=3, pre=3, prp=3, dropout=0.1):
        super().__init__(config)
        self.ple = ple
        self.plp = plp
        self.pre = pre
        self.prp = prp
        self.dim = 512
        self.b_emb = self.get_input_embeddings()
        self.emb = nn.Embedding(512, self.dim)
        # Commonsense features encoder
        self.encoder_layer_e = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_e = nn.TransformerEncoder(self.encoder_layer_e, num_layers=6)
        # Semantic features encoder
        self.encoder_layer_p = nn.TransformerEncoderLayer(d_model=2 * self.dim, nhead=8)
        self.tf_p = nn.TransformerEncoder(self.encoder_layer_p, num_layers=6)
        # Model for blend the two features
        self.bi_lstm = nn.LSTM(2 * self.dim, self.dim, 2, bidirectional=True, batch_first=True)

        # common ground
        self.cdim = 1792
        self.common_hidden = 1792
        # self.cdim = 3328
        # self.common_hidden = 3328
        self.cg_store = nn.LSTMCell(self.cdim, self.common_hidden)
        self.cg_affect = nn.LSTMCell(self.cdim, self.common_hidden)

        self.line_cr = nn.Linear(512, 512)
        self.line_crp = nn.Linear(1024, self.dim)
        self.line1 = nn.Linear(1024, 1024)
        self.line2 = nn.Linear(1024, 1024)
        self.line3 = nn.Linear(1024, 1024)

        self.lineu = nn.Linear(u_dim, 512)
        self.lineu2 = nn.Linear(u_dim, 256)
        self.linel = nn.Linear(l_dim, 256)
        self.linef = nn.Linear(f_dim, 256)
        self.linee = nn.Linear(1024, self.dim * (self.ple + self.pre))
        self.linep = nn.Linear(1024, self.dim * (self.plp + self.prp))
        self.global_c = nn.Linear(512, 1024)
        self.common_g = nn.Linear(1792, 1024)
        self.gelu = nn.GELU()
        self.smx = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        self.linenp = nn.Linear(1024, self.dim * (self.ple + self.pre + self.plp + self.prp))

    def forward(self,
                X,  # (batch, seq, dim=1024) features of utterance
                C,  # common ground
                r,  # (batch, seq, 3*768) common for listener
                cur_speakers,  # speaker name list
                lf,  # listener personal feature
                loc,  # utterance id
                input_ids=None,  # input
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,  # [CLS] -100 -100 label e(input_ids)
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # input_ids.shape[1] input length
        # print("input_ids.shape[1]:", input_ids.shape[1])
        ple = self.emb(torch.LongTensor([[i for i in range(1, self.ple + 1)]] * input_ids.shape[0]).cuda())  #(1, 3, 512)
        plp = self.emb(torch.LongTensor(
            [[i for i in range(self.ple + 1, self.ple + self.plp + 1)]] * input_ids.shape[0]).cuda())
        prp = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.pre - self.prp - 1,
                                                           input_ids.shape[1] - self.pre - 1)]] * input_ids.shape[0]
                                        ).cuda())
        pre = self.emb(torch.LongTensor([[i for i in range(input_ids.shape[1] - self.pre - 1, input_ids.shape[1] - 1)]]
                                        * input_ids.shape[0]).cuda())
        # hard prompt position
        pos = [i for i in range(1, self.ple + self.plp + 1)] + [i for i in range(input_ids.shape[1] - self.pre -
                                                                                 self.prp - 1, input_ids.shape[1] - 1)]
        # print("pos: ", pos)
        U = self.lineu(X)

        speaker_id = loc + 1
        now_speakers = cur_speakers[speaker_id]

        # common ground
        h_o = torch.zeros(self.common_hidden).cuda()
        c_o = torch.zeros(self.common_hidden).cuda()
        h_n = h_o
        c_n = c_o
        for i in range(speaker_id):
            h_o = h_n
            c_o = c_n
            h_store, c_store = self.cg_store(C[i], (h_o, c_o))
            h_affect, c_affect = self.cg_affect(C[i], (h_o, c_o))
            h_n = (cur_speakers[i] == now_speakers) * h_store + (cur_speakers[i] != now_speakers) * h_affect
            c_n = (cur_speakers[i] == now_speakers) * c_store + (cur_speakers[i] != now_speakers) * c_affect
        common_g = h_n
        common_g = self.common_g(common_g)  # 1024

        # personal feature
        LF = self.linef(lf)

        # modeling [p] + Semantic
        U = self.lineu2(X)  # 256
        # print("U:", U.shape)
        # print("LF:",LF.shape)
        p_p_1 = torch.cat((plp, prp), dim=1)
        p_p_2 = torch.cat((ple, pre), dim=1)
        p_p = torch.cat((p_p_1, p_p_2), dim=1) 
        com_r = self.line_cr(r)  
        com_r = torch.unsqueeze(com_r, 0)
        p_P = torch.cat((U, LF), dim=-1)
        p_P = p_P[:, loc - 1, :]  # (1, 512)
        p_P = torch.cat((p_P, com_r), dim=-1)
        p_P = self.tf_p(p_P)

        P = self.linenp(p_P).reshape(p_p.shape)
        p_p = torch.cat((P, p_p), dim=-1)

        # modeling for both
        p = p_p

        # soft prompt
        p = self.bi_lstm(p)[0]
        inputs_embeds = self.b_emb(input_ids)  # (batch,sentence_len,dim)
        inputs_embeds[:, pos, :] = p  #(1, 42, 4024)


        inputs_embeds[0][7] = common_g


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
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        mlm_out = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        logit = mlm_out.logits[:, - self.ple - self.plp - 2][:, [1372, 6161, 11956, 1313, 7974, 6378, 17437, 2490, 37018, 30883, 40788, 3915, 2430]]
        loss = mlm_out.loss
        return self.smx(logit), loss, mlm_out.logits