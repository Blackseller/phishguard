import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import dump

# Load the dataset
df = pd.read_csv("phishing_urls.csv")  # replace if different

# Show label types
print("Unique values in 'type':", df['type'].unique())

# Filter only 'phishing' and 'benign'
df = df[df['type'].isin(['phishing', 'benign'])]

# Convert labels to 1 and 0
df['label'] = df['type'].map({'phishing': 1, 'benign': 0})

X = df['url']
y = df['label']

# Create pipeline: TF-IDF + Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X, y)

# Save model
dump(pipeline, 'models/url_model.joblib')

print("✅ URL model trained and saved as 'url_model.joblib'")

