import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support
)



# Load test data
test_df = pd.read_csv("dataset/test_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])

y_true = test_df["label"]

# Load trained model (pipeline)
model = joblib.load("dataset/naive_bayes_ekman_model.pkl")

# Predict
y_pred = model.predict(test_df["text"])

#  Classification Report
print("=== Detailed Classification Report ===")
print(classification_report(y_true, y_pred, digits=2))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

# Performance Comparison with Benchmark (from Table 6)
benchmark = {
    'anger':    {'precision': 0.50, 'recall': 0.65, 'f1': 0.57},
    'disgust':  {'precision': 0.52, 'recall': 0.53, 'f1': 0.53},
    'fear':     {'precision': 0.61, 'recall': 0.76, 'f1': 0.68},
    'joy':      {'precision': 0.77, 'recall': 0.88, 'f1': 0.82},
    'neutral':  {'precision': 0.66, 'recall': 0.67, 'f1': 0.66},
    'sadness':  {'precision': 0.56, 'recall': 0.62, 'f1': 0.59},
    'surprise': {'precision': 0.53, 'recall': 0.70, 'f1': 0.61},
    'macro':    {'precision': 0.59, 'recall': 0.69, 'f1': 0.64}
}

# Compute current metrics
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=model.classes_, zero_division=0)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
micro_acc = accuracy_score(y_true, y_pred)

# Prepare comparison DataFrame
comparison = pd.DataFrame({
    'Emotion': model.classes_,
    'My Precision': p,
    'Benchmark Precision': [benchmark[e]['precision'] for e in model.classes_],
    'My Recall': r,
    'Benchmark Recall': [benchmark[e]['recall'] for e in model.classes_],
    'My F1': f1,
    'Benchmark F1': [benchmark[e]['f1'] for e in model.classes_]
})

# Add macro row
comparison.loc[len(comparison.index)] = [
    'macro', macro_p, benchmark['macro']['precision'],
    macro_r, benchmark['macro']['recall'],
    macro_f1, benchmark['macro']['f1']
]

print("\n=== Performance Comparison ===")
print(comparison.round(2).to_string(index=False))
print(f"\nMicro-Average Accuracy: {micro_acc:.2f}")
print(f"Macro-Average F1:       {macro_f1:.2f}")





# Vocabulary Pruning Analysis
print("\n=== Vocabulary Pruning Impact ===")
vectorizer = model.named_steps['vectorizer']
original_vocab_size = len(vectorizer.vocabulary_)

for min_df in [1, 2, 5, 10]:
    pruned_vectorizer = CountVectorizer(min_df=min_df)
    X_pruned = pruned_vectorizer.fit_transform(test_df["text"])
    vocab_size = len(pruned_vectorizer.vocabulary_)
    reduction = 100 * (original_vocab_size - vocab_size) / original_vocab_size
    print(f"min_df={min_df:<2} → vocab size: {vocab_size:5} (↓ {reduction:.1f}%)")

# Text Normalization Impact Analysis
def analyze_cleaning_impact():
    
    
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '<NUM>', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]
        return ' '.join(tokens)

    cleaned_texts = test_df["text"].apply(clean_text)
    cleaned_preds = model.predict(cleaned_texts)
    
    print("\n=== Text Normalization Impact ===")
    print(classification_report(y_true, cleaned_preds, digits=2))
    _, _, cleaned_f1, _ = precision_recall_fscore_support(y_true, cleaned_preds, average='macro')
    print(f"Macro-Average F1 after cleaning: {cleaned_f1:.2f}")

analyze_cleaning_impact()
