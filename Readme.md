
# AC-EIC: Addressee-Centered Emotion Inference in Conversations

**AC-EIC** is a prompt-based framework for **Emotion Inference in Conversations (EIC)**, aiming to infer an **addressee’s emotional reaction** to the dialogue context **without observing the addressee’s future utterances**. In contrast to standard Emotion Recognition in Conversation (ERC), EIC is inherently more challenging due to the missing target response. AC-EIC addresses this challenge by explicitly centering the addressee and enhancing dialogue understanding with **commonsense knowledge** and **addressee personality-driven emotional knowledge**.



---

## Overview

The emotional reactions of users to the dialogue context can guide the dialogue system to generate more satisfactory responses. Compared to the traditional task of Emotion Recognition in Conversation (ERC), the task of Emotion Inference in Conversations (EIC) is more challenging as it aims to infer the addressee’s emotional reactions to the context when the addressee’s utterances are unknown. Previous studies on EIC mainly focus on dialogue history information, neglecting the crucial role of the addressee as the subject of in emotion inference. In this paper, we propose an Addressee-Centered Emotion Inference in Conversations (AC-EIC) method, which can understand the dialogue history supplemented by commonsense knowledge and emotional knowledge based on the addressee’s personality. Additionally, due to the scarcity of character personality data, we manually collect the personality information of characters from three commonly used EIC datasets, expanding the original dialogue dataset. The experimental results show that AC-EIC achieves the new state-of-the-art performance on multiple datasets, demonstrating that our method can make more accurate inferences by focusing more on the addressee. Additionally, we also found that the mixed use of different types of knowledge has a positive impact on EIC tasks.

---



## Repository Structure

A suggested clean layout:

```text
src/
  configs/          
  data/             
  features/         
  models/          
  training/        
  scripts/        
````





## Installation

### Requirements

* Python >= 3.9
* PyTorch >= 1.13
* transformers
* scikit-learn
* pandas, numpy

### Setup

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# Create env (conda example)
conda create -n ac_eic python=3.10 -y
conda activate ac_eic

pip install -r requirements.txt
```



## Training & Evaluation

### Quick Start

If you have a script entrypoint (recommended):

```bash
bash run.sh
```


## Citation

If you find this repository helpful, please cite:

```bibtex
@article{xu2025ac,
  title={AC-EIC: addressee-centered emotion inference in conversations},
  author={Xu, Xingle and Feng, Shi and Cui, Yuan and Zhang, Yifei and Wang, Daling},
  journal={International Journal of Machine Learning and Cybernetics},
  pages={1--18},
  year={2025},
  publisher={Springer}
}
```




