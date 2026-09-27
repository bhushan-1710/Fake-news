"""
train_model.py
--------------
This script trains a Logistic Regression model on the news dataset
using TF-IDF vectorization, evaluates it, and saves the trained
model and vectorizer to the model/ directory.

Run this script ONCE before starting the Flask app:
    python train_model.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib


def main():
    # ── 1. Load dataset ──────────────────────────────────────────────
    dataset_path = os.path.join("dataset", "news.csv")

    if not os.path.exists(dataset_path):
        print("ERROR: Dataset file not found at", dataset_path)
        print("Please place news.csv in the dataset/ folder.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"  Total rows loaded: {len(df)}")

    # ── 2. Preprocessing ─────────────────────────────────────────────
    # Remove rows with missing values
    df = df.dropna(subset=["text", "label"])
    print(f"  Rows after removing missing values: {len(df)}")

    # Convert text to lowercase
    df["text"] = df["text"].str.lower()

    # Separate features and labels
    X = df["text"]
    y = df["label"]

    print(f"  Label distribution:\n{y.value_counts().to_string()}")
    print()

    # ── 3. Split into training and testing sets (80/20) ───────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    print()

    # ── 4. TF-IDF Vectorization ──────────────────────────────────────
    print("Applying TF-IDF vectorization (max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000)

    # Fit on training data only, then transform both sets
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")
    print()

    # ── 5. Train Logistic Regression ─────────────────────────────────
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("  Training complete!")
    print()

    # ── 6. Evaluate the model ────────────────────────────────────────
    print("Evaluating model on test data...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="FAKE")
    recall = recall_score(y_test, y_pred, pos_label="FAKE")
    f1 = f1_score(y_test, y_pred, pos_label="FAKE")

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ── 7. Save model and vectorizer ─────────────────────────────────
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Model saved to:      {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print()
    print("Done! You can now run the Flask app with: python app.py")


if __name__ == "__main__":
    main()
