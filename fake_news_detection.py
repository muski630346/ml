import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# =======================
# 1. Dataset Preparation
# =======================
# Check if merged dataset already exists
if not os.path.exists("fake_news.csv"):
    print("Preparing dataset...")

    # Load individual files with proper paths (using raw string)
    fake = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\New folder\New folder\archive (7)\Fake.csv")
    true = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\New folder\New folder\archive (7)\True.csv")

    # Add labels: 1 for Fake, 0 for Real
    fake['label'] = 1
    true['label'] = 0

    # Merge datasets
    df = pd.concat([fake, true], ignore_index=True)
    df = df[['title', 'text', 'label']]  # Keep only necessary columns
    df.to_csv("fake_news.csv", index=False)
    print("Dataset saved as 'fake_news.csv'.")
else:
    print("Using existing 'fake_news.csv' file.")

# =======================
# 2. Model Training
# =======================
print("Training model...")

# Load and preprocess data
df = pd.read_csv('fake_news.csv')
df['text'] = df['text'].fillna('')

X = df['text']
y = df['label']

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved.")

# =======================
# 3. Dynamic Prediction
# =======================
print("\n📰 Fake News Detection System (type 'exit' to quit)\n")

# Load model & vectorizer
model = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

while True:
    user_input = input("Enter a news article (or type 'exit' to quit):\n> ")
    if user_input.lower() == "exit":
        break
    if not user_input.strip():
        print("⚠️ Please enter some text.\n")
        continue

    user_input_tfidf = tfidf.transform([user_input])
    prediction = model.predict(user_input_tfidf)
    proba = model.predict_proba(user_input_tfidf)[0]
    confidence = max(proba) * 100

    print(f"\n🔍 Prediction: {'FAKE NEWS ❌' if prediction[0] == 1 else 'REAL NEWS ✅'} ({confidence:.2f}% confidence)\n")
