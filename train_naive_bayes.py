# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load the datasets
# train_df = pd.read_csv("dataset/train_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])
# dev_df = pd.read_csv("dataset/dev_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])
# test_df = pd.read_csv("dataset/test_ekman.tsv", sep="\t", skiprows=1, names=["text", "label"])

# # Define pipeline: vectorization + Naive Bayes model
# model = Pipeline([
#     ('vectorizer', CountVectorizer()),  # Bag-of-Words vectorizer
#     ('classifier', MultinomialNB())     # Naive Bayes classifier
# ])

# # Train the model
# model.fit(train_df["text"], train_df["label"])

# # Evaluate on the dev set
# dev_preds = model.predict(dev_df["text"])
# print("Validation Accuracy:", accuracy_score(dev_df["label"], dev_preds))
# print("Validation Classification Report:\n", classification_report(dev_df["label"], dev_preds))

# # Save the model
# joblib.dump(model, "dataset/naive_bayes_ekman_model.pkl")

# # Optional: Evaluate on test set
# test_preds = model.predict(test_df["text"])
# print("Test Accuracy:", accuracy_score(test_df["label"], test_preds))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # Changed from CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
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
        ngram_range=(1, 2),  # Use unigrams and bigrams
        max_features=10000,   # Limit vocabulary size
        stop_words='english'  # Remove common words
    )),
    ('classifier', MultinomialNB(alpha=0.1))  # Add smoothing
])

# Train with class weights
model.fit(train_df["text"], train_df["label"], 
          classifier__sample_weight=sample_weights)

# Evaluation function
def evaluate(model, df, name):
    preds = model.predict(df["text"])
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(df["label"], preds))
    print(classification_report(df["label"], preds, zero_division=0))

# Evaluate
evaluate(model, dev_df, "Validation")
evaluate(model, test_df, "Test")

# Save model
joblib.dump(model, "dataset/naive_bayes_ekman_model.pkl")