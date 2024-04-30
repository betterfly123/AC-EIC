# AC-EIC: Addressee-Centered Emotion Inference in Conversations

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

# Instruction Example
1.
instruction: Complete the emotion inference task by predicting the emotion of the next utterance using the given conversation history, the emotion can only be chosen from [happy, grateful, relaxed, positive-other, neutral, anger, sadness, fear, depress, disgust, astonished, worried, negative-other], note that the emotion of the next utterance is unknown.

input: What a coincidence The car is all right It's you. It's okay. It's okay. Thank you

output: neutral

2.
instruction: Complete the emotion inference task by predicting the emotion of the next utterance using the given conversation history, the emotion can only be chosen from [neutral, surprise, fear, sadness, joy, disgust, anger], note that the emotion of the next utterance is unknown.

input: I'm supposed to attach a brackety thing to the side things, using a bunch of these little worm guys. I have no brackety thing, I see no whim guys whatsoever and- I cannot feel my legs. I'm thinking we've got a bookcase here. It's a beautiful thing. What's this? Which goes where? I have no idea. Done with the bookcase! All finished! This was Carol's favorite beer. She always drank it out of the can, I should have known. Yes, please don't spoil all this fun. You guys. Oh, God. You got screwed.

output: Mad

# Requirements
* python 3.8
* cuda 11.7
* pytorch 2.0.0
* numpy 1.22.3
* pandas 1.4.4
* transformers 4.18.0
* scikit-learn 1.1.2
# Preparation
