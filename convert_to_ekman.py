import pandas as pd
import json
import os
import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required resources for tokenization and lemmatization
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# === Step 1: Load Ekman mapping (fine-grained -> Ekman classes) ===
with open("dataset/ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

# Create reverse mapping: fine-grained emotion -> Ekman category
fine_to_ekman = {fine: ekman for ekman, fine_list in ekman_mapping.items() for fine in fine_list}
fine_to_ekman["neutral"] = "neutral"  # Include neutral explicitly

# === Step 2: Load emotions.txt to map label IDs -> fine-grained emotion names ===
with open("dataset/emotions.txt", "r") as f:
    id_to_label = {str(i): label.strip() for i, label in enumerate(f)}

# === Step 3: Function to process and clean one dataset (train/dev/test) ===
def process_file(filepath):
    df = pd.read_csv(filepath, sep="\t", header=None, names=["text", "labels", "ids"])

    # Ignore the 3rd column (ids)
    df = df[["text", "labels"]]

    # Remove multi-label samples (those with comma-separated label IDs)
    df = df[df["labels"].apply(lambda x: "," not in str(x))]

    # Convert label ID to fine-grained label
    df["labels"] = df["labels"].astype(str).map(id_to_label)

    # Drop rows with unmapped labels
    df = df.dropna(subset=["labels"])

    # Convert fine-grained label to Ekman class
    df["labels"] = df["labels"].map(lambda label: fine_to_ekman.get(label, None))

    # Drop rows with unmapped emotions (i.e. not in 6 Ekman + neutral)
    df = df.dropna(subset=["labels"])

    return df

# === Step 4: Clean and tokenize text ===
def clean_and_tokenize(text):
    # Lowercase
    text = text.lower()

    # Replace numbers with a placeholder
    text = re.sub(r"\d+", "<NUM>", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize each token
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# === Step 5: Frequency analysis ===
def analyze_frequencies(df, split_name):
    all_tokens = []
    for text in df["text"]:
        tokens = clean_and_tokenize(text)
        all_tokens.extend(tokens)

    # Count unigrams and bigrams
    unigram_counts = Counter(all_tokens)
    bigram_counts = Counter(zip(all_tokens, all_tokens[1:]))

    # Print top 10
    print(f"\n==== {split_name.upper()} DATASET ====")
    print("Top 10 unigrams:")
    print(unigram_counts.most_common(10))

    print("\nTop 10 bigrams:")
    print(bigram_counts.most_common(10))

# === Step 6: Process all splits (train/dev/test) and save cleaned versions ===
splits = ["train", "dev", "test"]
processed_data = {}

for split in splits:
    path = f"dataset/{split}.tsv"
    df = process_file(path)

    # Save cleaned Ekman-mapped dataset
    out_path = f"dataset/{split}_ekman.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"{split}_ekman.tsv saved with {len(df)} rows.")

    # Save processed df to run analysis later
    processed_data[split] = df

# === Step 7: Run shallow analysis ===
for split, df in processed_data.items():
    analyze_frequencies(df, split)
