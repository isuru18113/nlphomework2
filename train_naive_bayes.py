import pandas as pd
from numpy import std

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# Load datasets
train_df = pd.read_csv("dataset/train_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])
dev_df = pd.read_csv("dataset/dev_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])
test_df = pd.read_csv("dataset/test_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])

# Calculate class weights
sample_weights = compute_sample_weight('balanced', train_df["label"])

# Improved pipeline
model = Pipeline([
    ('vectorizer', TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams and bigrams
        max_features=10000,   # Limit vocabulary size
        stop_words='english'  # Remove common words
    )),
    ('classifier', MultinomialNB(alpha=0.1))  
])

# Train with class weights
model.fit(train_df["text"], train_df["label"], 
          classifier__sample_weight=sample_weights)

#Evaluation function
def evaluate(model, df, name):
    preds = model.predict(df["text"])
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(df["label"], preds))
    print(classification_report(df["label"], preds, zero_division=0))

#Evaluate
evaluate(model, dev_df, "Validation")
evaluate(model, test_df, "Test")


# Save model
joblib.dump(model, "dataset/naive_bayes_ekman_model.pkl")
