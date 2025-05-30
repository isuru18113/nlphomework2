NLP Assignment #2 /
Isuru Bandara /
ii22012
---

## Token Frequency Statistics

## Results of Convert_to_ekman.py

### 📊 TRAIN DATASET

#### Top 10 Unigrams

| Rank | Unigram | Frequency |
| ---- | ------- | --------- |
| 1    | the     | 14,896    |
| 2    | i       | 13,966    |
| 3    | a       | 12,047    |
| 4    | to      | 10,442    |
| 5    | it      | 9,026     |
| 6    | you     | 8,203     |
| 7    | and     | 7,331     |
| 8    | that    | 7,044     |
| 9    | is      | 6,991     |
| 10   | name    | 6,791     |

#### Top 10 Bigrams

| Rank | Bigram        | Frequency |
| ---- | ------------- | --------- |
| 1    | (’ , s)       | 1,805     |
| 2    | (’ , t)       | 1,609     |
| 3    | (in , the)    | 1,159     |
| 4    | (i , ’)       | 1,141     |
| 5    | (this , is)   | 962       |
| 6    | (of , the)    | 934       |
| 7    | (to , be)     | 800       |
| 8    | (’ , m)       | 744       |
| 9    | (thank , you) | 699       |
| 10   | (for , the)   | 685       |

---

### DEV DATASET

#### Top 10 Unigrams

| Rank | Unigram | Frequency |
| ---- | ------- | --------- |
| 1    | the     | 1,884     |
| 2    | i       | 1,664     |
| 3    | a       | 1,500     |
| 4    | to      | 1,251     |
| 5    | it      | 1,208     |
| 6    | you     | 1,031     |
| 7    | and     | 939       |
| 8    | that    | 857       |
| 9    | name    | 856       |
| 10   | is      | 844       |

#### Top 10 Bigrams

| Rank | Bigram      | Frequency |
| ---- | ----------- | --------- |
| 1    | (’ , s)     | 244       |
| 2    | (’ , t)     | 199       |
| 3    | (i , ’)     | 148       |
| 4    | (in , the)  | 144       |
| 5    | (this , is) | 125       |
| 6    | (it , ’)    | 109       |
| 7    | (of , the)  | 103       |
| 8    | (’ , m)     | 96        |
| 9    | (to , be)   | 88        |
| 10   | (i , love)  | 87        |

---

### TEST DATASET

#### Top 10 Unigrams

| Rank | Unigram | Frequency |
| ---- | ------- | --------- |
| 1    | i       | 1,775     |
| 2    | the     | 1,767     |
| 3    | a       | 1,534     |
| 4    | to      | 1,270     |
| 5    | it      | 1,190     |
| 6    | you     | 1,073     |
| 7    | and     | 931       |
| 8    | name    | 870       |
| 9    | that    | 869       |
| 10   | is      | 848       |

#### Top 10 Bigrams

| Rank | Bigram        | Frequency |
| ---- | ------------- | --------- |
| 1    | (’ , s)       | 249       |
| 2    | (’ , t)       | 203       |
| 3    | (i , ’)       | 158       |
| 4    | (in , the)    | 156       |
| 5    | (of , the)    | 129       |
| 6    | (this , is)   | 116       |
| 7    | (’ , m)       | 111       |
| 8    | (it , ’)      | 104       |
| 9    | (thank , you) | 103       |
| 10   | (i , love)    | 87        |

---

## multi-class Naïve Bayes classifier

### Validation Set Results

**Accuracy**: `0.444`

#### 📋 Classification Report

| Emotion  | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| anger    | 0.29      | 0.39   | 0.33     | 485     |
| disgust  | 0.08      | 0.26   | 0.12     | 61      |
| fear     | 0.15      | 0.47   | 0.23     | 66      |
| joy      | 0.73      | 0.58   | 0.65     | 1668    |
| neutral  | 0.50      | 0.33   | 0.40     | 1592    |
| sadness  | 0.25      | 0.51   | 0.34     | 241     |
| surprise | 0.26      | 0.37   | 0.31     | 435     |

**Overall Metrics**:

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 0.444 |
| Macro Avg F1    | 0.34  |
| Weighted Avg F1 | 0.46  |

---

###  Test Set Results

**Accuracy**: `0.446`

#### 📋 Classification Report

| Emotion  | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| anger    | 0.31      | 0.40   | 0.35     | 520     |
| disgust  | 0.14      | 0.39   | 0.20     | 76      |
| fear     | 0.16      | 0.44   | 0.24     | 77      |
| joy      | 0.71      | 0.57   | 0.63     | 1603    |
| neutral  | 0.53      | 0.35   | 0.42     | 1606    |
| sadness  | 0.25      | 0.46   | 0.33     | 259     |
| surprise | 0.26      | 0.39   | 0.31     | 449     |

**Overall Metrics**:

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 0.446 |
| Macro Avg F1    | 0.36  |
| Weighted Avg F1 | 0.47  |


---
## Evaluation

### Confusion Matrix

![Confusion Matrix](https://github.com/isuru18113/nlphomework2/raw/main/confusion_matrix.png)

---
## 📊 Results Analysis

### Impact of Vocabulary Pruning

| `min_df` | Vocabulary Size | Reduction | Observations                                                   |
| -------- | --------------- | --------- | -------------------------------------------------------------- |
| 1        | 7,766           | –         | Baseline vocabulary                                            |
| 2        | 3,065           | ↓ 69.3%   | Significant reduction; balances size and information           |
| 5        | 1,187           | ↓ 88.1%   | Aggressive pruning; some useful rare terms may be lost         |
| 10       | 599             | ↓ 94.0%   | Very limited vocabulary; likely to hurt classifier performance |

**Conclusion**:
Moderate pruning (`min_df=2`) reduces noise and training time without major loss of accuracy. Higher thresholds risk discarding important emotional cues.

---

###  Effect of Text Normalization

| Preprocessing Method | Macro F1 Score |
| -------------------- | -------------- |
| Raw Text             | **0.36**       |
| Cleaned Text         | **0.34**       |

**Conclusion**:
Text cleaning (lemmatization, lowercasing, removing numbers/punctuation) **slightly decreased performance**. This suggests that some of the raw stylistic or lexical variations may carry emotional signals that cleaning removes.

---



### Class-wise Performance

| Emotion  | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| Joy      | 0.71      | 0.57   | 0.63     |
| Neutral  | 0.53      | 0.35   | 0.42     |
| Sadness  | 0.25      | 0.46   | 0.33     |
| Disgust  | 0.14      | 0.39   | 0.20     |
| Anger    | 0.26      | 0.24   | 0.25     |
| Surprise | 0.24      | 0.24   | 0.24     |
| Fear     | 0.21      | 0.24   | 0.23     |

* **Joy and Neutral** have the highest F1 scores — likely due to larger representation and clearer lexical cues.
* **Disgust, Anger, and Fear** are very low and perform poorly.

---

### Performance Overview

| Metric         | Value |
| -------------- | ----- |
| Micro Accuracy | 0.45  |
| Macro F1 Score | 0.36  |
| Avg Precision  | 0.33  |
| Avg Recall     | 0.35  |

**Observation**:
The gap between micro and macro metrics highlights class imbalance. The model is biased toward frequent classes.

---

### evaluate_classifier.py Results

---

## 📊 Detailed Classification Report (Test Set)

**Accuracy**: `0.45`
**Macro-Average F1**: `0.36`
**Confusion Matrix**: `confusion_matrix.png`

| Emotion  | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| anger    | 0.31      | 0.40   | 0.35     | 520     |
| disgust  | 0.14      | 0.39   | 0.20     | 76      |
| fear     | 0.16      | 0.44   | 0.24     | 77      |
| joy      | 0.71      | 0.57   | 0.63     | 1603    |
| neutral  | 0.53      | 0.35   | 0.42     | 1606    |
| sadness  | 0.25      | 0.46   | 0.33     | 259     |
| surprise | 0.26      | 0.39   | 0.31     | 449     |

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 0.45  |
| Macro Avg F1    | 0.36  |
| Weighted Avg F1 | 0.47  |

---

## Performance Comparison with Benchmark

| Emotion   | My Precision | Benchmark Precision | My Recall | Benchmark Recall | My F1    | Benchmark F1 |
| --------- | ------------ | ------------------- | --------- | ---------------- | -------- | ------------ |
| anger     | 0.31         | 0.50                | 0.40      | 0.65             | 0.35     | 0.57         |
| disgust   | 0.14         | 0.52                | 0.39      | 0.53             | 0.20     | 0.53         |
| fear      | 0.16         | 0.61                | 0.44      | 0.76             | 0.24     | 0.68         |
| joy       | 0.71         | 0.77                | 0.57      | 0.88             | 0.63     | 0.82         |
| neutral   | 0.53         | 0.66                | 0.35      | 0.67             | 0.42     | 0.66         |
| sadness   | 0.25         | 0.56                | 0.46      | 0.62             | 0.33     | 0.59         |
| surprise  | 0.26         | 0.53                | 0.39      | 0.70             | 0.31     | 0.61         |
| **Macro** | **0.34**     | **0.59**            | **0.43**  | **0.69**         | **0.36** | **0.64**     |

* **Micro-Average Accuracy**: `0.45`
* **Macro-Average F1**: `0.36`

---

## Vocabulary Pruning Impact

| `min_df` | Vocab Size | Reduction |
| -------- | ---------- | --------- |
| 1        | 7766       | ↓ 22.3%   |
| 2        | 3065       | ↓ 69.3%   |
| 5        | 1187       | ↓ 88.1%   |
| 10       | 599        | ↓ 94.0%   |

---

## Text Normalization Impact

**Accuracy After Cleaning**: `0.43`
**Macro-Average F1 After Cleaning**: `0.34`

| Emotion  | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| anger    | 0.30      | 0.36   | 0.33     | 520     |
| disgust  | 0.10      | 0.38   | 0.16     | 76      |
| fear     | 0.17      | 0.43   | 0.24     | 77      |
| joy      | 0.70      | 0.56   | 0.62     | 1603    |
| neutral  | 0.51      | 0.32   | 0.39     | 1606    |
| sadness  | 0.25      | 0.45   | 0.32     | 259     |
| surprise | 0.24      | 0.40   | 0.30     | 449     |

---



