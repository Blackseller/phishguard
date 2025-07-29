# models/url_model.joblib र models/email_model.joblib को लागि डमी मोडेल तयार गर्ने
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
import os

# डाइरेक्ट्री बनाउने
os.makedirs("models", exist_ok=True)

# डमी डाटा जनरेट गर्ने
X, y = make_classification(n_samples=100, n_features=20)

# URL मोडेल तयार गर्ने
url_model = RandomForestClassifier()
url_model.fit(X, y)
joblib.dump(url_model, "url_model.joblib")

# Email मोडेल तयार गर्ने
email_model = RandomForestClassifier()
email_model.fit(X, y)
joblib.dump(email_model, "email_model.joblib")