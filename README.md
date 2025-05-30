NLP Assignment #2
Isuru Bandara
ii22012

## Evaluation


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
