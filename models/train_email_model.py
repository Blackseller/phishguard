import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

# Load dataset
df = pd.read_csv("phishing_emails.csv")  # Replace with your actual file name if different

# Use 'body' for email text, and 'label' for phishing (1) or legitimate (0)
X = df['body']
y = df['label']

# Create a pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))  # increase if needed
])

# Train the model
pipeline.fit(X, y)

# Save model to models/ directory
dump(pipeline, 'email_model.joblib')

print("✅ Email model trained and saved as 'email_model.joblib'")
