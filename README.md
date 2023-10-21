# Foundation of Dialogue: Prompting Common Ground for Emotion Inference in Conversations

# Dataset category distributions.

| MELD  | Neutral | Surprise | Fear | Sadness |  Joy  | Disgust | Anger | Total |
|-------|:-------:|:--------:|:----:|:-------:|:-----:|:-------:|:-----:|:-----:|
| Train |  4,710  |   1,205  |  268 |   683   | 1,743 |   271   | 1,109 | 9,989 |
| Val   |   470   |    150   |  40  |   111   |  163  |    22   |  153  | 1,109 |
| Test  |  1,256  |    281   |  50  |   208   |  402  |    68   |  345  | 2,610 |

| EmoryNLP | Joyful | Mad | Peaceful | Neutral | Sad | Powerful | Scared | Total |
|----------|:------:|:---:|:--------:|:-------:|:---:|:--------:|:------:|:-----:|
| Train    |  1,677 | 785 |    638   |  2,485  | 474 |    551   |   941  | 7,551 |
| Val      |   205  |  97 |    82    |   322   |  51 |    70    |   127  |  954  |
| Test     |   217  |  86 |    111   |   288   |  70 |    96    |   116  |  984  |


| CPED  | happy | grateful | relaxed | positive-other | neutral |  anger | sadness |  fear | depress | disgust | astonished | worried | negative-other |  Total |
|-------|:-----:|----------|---------|----------------|---------|:------:|---------|:-----:|:-------:|---------|:----------:|:-------:|:--------------:|:------:|
| Train | 3,547 |    206   |  2,150  |      74,67     |  31,758 | 13,400 |  2,217  | 1,980 |  9,817  |  1,353  |    2,430   |  6,142  |      9,881     | 94,188 |
| Val   |  370  |    13    |   517   |       792      |  3,126  |  1,608 |   274   |  117  |  1,446  |   198   |     313    |   661   |      1,702     | 11,138 |
| Test  | 2,604 |    31    |  2,150  |      1,685     |  7,991  |  3,031 |   530   |  872  |  2,792  |   435   |    1,433   |  1,489  |      2,395     | 27,439 |

# Case Study

| context                                                                     | Phoebe: Oh my God, oh my God!<br/>Poor Monica! (to bless me) Chandler: What, what, what?! (to listen to personx)<br/>Phoebe: What?! He was with her when he wrote this poem. (to read the poem)<br/>Phoebe: Look, ``My vessel so empty with nothing inside. Now that I've touched you, you seem emptier still." (to learn about persony)<br/>*Phoebe:                                                                                                            |
|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| personality                                                                 | Brutal Honesty: She can be quite blunt to her friends to the point that she comes across as rude.<br/>Cannot Keep a Secret: Especially when it comes to Monica's secrets.<br/>Beware the Silly Ones: She may seem like a sweet Granola Girl with a kooky sense of humor, but she was tough enough to mug older boys when she was just a teen living on the street.<br/>(Due to space constraints, only part of the personality information is given) |
| CICERO                                                                      | Phoebe is curious to hear the speaker's reaction.                                                                                                                                                                                                                                                                                                                                                                                                    |
| DialogInfer-Ensemble<br/>DialogInfer-(S+G)+K<br/>DialogueGLP(F+U)<br/>PCGEI | Neutral<br/>Neutral<br/>Neutral<br/>Surprised√                                                                                                                                                                                                                                                                                                                                                                                                       |
| label                                                                       | Phoebe: Oh my God, oh my God! Poor Monica! (Surprise)<br/>Chandler: What, what, what?! (Surprise)<br/>Phoebe: What?! He was with her when he wrote this poem. (Neutral)<br/>Phoebe: Look, ``My vessel so empty with nothing inside. Now that I've touched you, you seem emptier still." (Neutral)<br/>*Phoebe: He thinks Monica is empty, she is the empty vase! (Surprise)                                                                                          |

Table shows an example of EIC from the MELD dataset and the results of the different models for emotion inference. The background of the example is that a man writes a poem to Monica, and Phoebe temporarily saves it for Monica. Subsequently, Phoebe and Chandler engage in a conversation regarding the poem. Based on the labels provided in the table, it is evident that the emotion of the addressee, Phoebe, remains "Neutral" in two consecutive utterances prior to the target utterance. However, the emotion suddenly changes to "Surprise'' for the target utterance. Instances of dialogues characterized by abrupt emotional transitions have the potential to misguide the model and exacerbate the challenges in emotion inference. Furthermore, the addressee Phoebe's utterance "He thinks Monica is empty, she is the empty vase!" and the emotion "Surprise" are unknown.

The experimental results indicate that the existing strong baseline models predict the addressee Phoebe's emotion as "Neutral". The analysis reveals that the strong baseline models fail to accurately account for the impact of emotion persistence. It simply continues the pattern of emotion persistence and predicts a "Neutral" emotion, thereby overlooking abrupt emotional transitions. Our model PCGEI comprehends the context by integrating each utterance with relevant commonsense knowledge. Simultaneously, the model also incorporates the addressee's personality information. The model's advanced comprehension of the complete dialogue enables it to effectively manage the unfavorable impacts of emotion persistence. Given Phoebe's "Brutal Honesty" personality trait, it is certain that she will openly recite the poem given by someone else to Monica. Additionally, Phoebe struggles with keeping secrets, particularly when it involves Monica. Emotional knowledge generated by CICERO, which states "Phoebe is curious to see Monica's empty vessel" implies that Phoebe is about to express her opinion regarding the content of the letter. Context, persona and knowledge establish the foundation of the dialogue and serve as a premise to infer Phoebe's emotion. Clearly, PCGEI can infer that Phoebe will be surprised to find out and blurt out-``He thinks Monica is empty, she is the empty vase!".

In contrast, the strong baseline models lack the capability to acquire such comprehensive information for modeling. It can only make conventional inferences strictly based on the context, without possessing a profound understanding of the entire conversation to ensure accurate results. The results of our study revealed that our model exhibits superior conversational comprehension, thus demonstrating the essentiality and validity of incorporating the foundation of dialogue in the EIC task.


# Requirements
* python 3.8
* cuda 11.7
* pytorch 2.0.0
* numpy 1.22.3
* pandas 1.4.4
* transformers 4.18.0
* scikit-learn 1.1.2
# Preparation
