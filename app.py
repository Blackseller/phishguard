# -*- coding: utf-8 -*-
"""
Advanced Phishing Detection Engine v5.0
Author: Security Analytics Team
Date: 2024-07-23
"""

import re
import os
import socket
import ssl
import ipaddress
import logging
from urllib.parse import urlparse
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import whois
import tldextract
import numpy as np
import joblib
import pandas as pd
import dns.resolver
from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS
from flask_wtf import FlaskForm, CSRFProtect
from flask_wtf.csrf import generate_csrf
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired
from wtforms.csrf.session import SessionCSRF
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
import hashlib
import json
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import time
import threading
from sklearn.feature_extraction.text import TfidfVectorizer

# Get absolute path to current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PhishingDetector')

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_SECRET_KEY'] = os.environ.get('CSRF_SECRET_KEY', 'anothersecretkey')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
csrf = CSRFProtect(app)

# Initialize models
MODEL_DIR = os.path.join(BASE_DIR, "models")
url_model = None
email_model = None
similarity_model = None
anomaly_detector = None
phishing_corpus = []
brand_embeddings = {}
vectorizer = None
phishing_keywords = None
nlp_pipeline = None

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logger.info(f"Created directory: {MODEL_DIR}")

# Create default files if they don't exist
phishing_corpus_file = os.path.join(MODEL_DIR, "phishing_corpus.json")
if not os.path.exists(phishing_corpus_file):
    default_corpus = ["login", "verify", "account", "security", "update", "urgent", "suspended", "limited", "action", "required"]
    with open(phishing_corpus_file, "w") as f:
        json.dump(default_corpus, f)
    logger.info("Created default phishing_corpus.json")

phishing_keywords_file = os.path.join(MODEL_DIR, "phishing_keywords.txt")
if not os.path.exists(phishing_keywords_file):
    default_keywords = ["password", "ssn", "credit card", "login", "verify", "account", "credentials", "pin", "security code"]
    with open(phishing_keywords_file, "w") as f:
        f.write("\n".join(default_keywords))
    logger.info("Created default phishing_keywords.txt")

# Initialize NLP models
try:
    logger.info(f"Loading models from: {MODEL_DIR}")
    
    # Load URL model
    url_model_path = os.path.join(MODEL_DIR, "url_model.joblib")
    if os.path.exists(url_model_path):
        url_model = joblib.load(url_model_path)
        logger.info("URL model loaded successfully")
    else:
        logger.warning(f"URL model not found at {url_model_path}. Using rule-based scoring.")
    
    # Load email model
    email_model_path = os.path.join(MODEL_DIR, "email_model.joblib")
    if os.path.exists(email_model_path):
        email_model = joblib.load(email_model_path)
        logger.info("Email model loaded successfully")
    else:
        logger.warning(f"Email model not found at {email_model_path}. Using rule-based scoring.")
    
    # Load similarity model
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Similarity model loaded")
    
    # Initialize NLP pipeline
    try:
        nlp_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("NLP pipeline initialized")
    except Exception as e:
        logger.warning(f"Could not initialize NLP pipeline: {e}")
        nlp_pipeline = None
    
    # Initialize anomaly detector
    anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    logger.info("Anomaly detector initialized")
    
    # Train anomaly detector
    logger.info("Training anomaly detector...")
    X_train = np.random.rand(500, 10) * 100
    anomaly_detector.fit(X_train)
    logger.info("Anomaly detector trained")
    
    # Load phishing corpus
    try:
        with open(phishing_corpus_file, "r") as f:
            phishing_corpus = json.load(f)
        logger.info(f"Loaded {len(phishing_corpus)} phishing patterns")
    except Exception as e:
        logger.warning(f"Could not load phishing corpus: {e}")
        phishing_corpus = ["login", "verify", "account", "security", "update", "urgent"]
    
    # Load phishing keywords
    try:
        with open(phishing_keywords_file, "r") as f:
            phishing_keywords = [line.strip().lower() for line in f.readlines()]
        logger.info(f"Loaded {len(phishing_keywords)} phishing keywords")
    except Exception as e:
        logger.warning(f"Could not load phishing keywords: {e}")
        phishing_keywords = ["password", "ssn", "credit card", "login", "verify", "account"]
    
    # Precompute brand embeddings
    brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'bankofamerica', 
              'netflix', 'ebay', 'wellsfargo', 'chase', 'citibank', 'dropbox', 
              'linkedin', 'twitter', 'facebook', 'instagram']
    brand_embeddings = {brand: similarity_model.encode(brand) for brand in brands}
    logger.info("Brand embeddings computed")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    
    # Train on sample data
    sample_texts = ["login to your account", "verify your identity", 
                   "security alert", "suspicious activity", 
                   "free shipping offer", "newsletter subscription"]
    vectorizer.fit(sample_texts)
    
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}", exc_info=True)
    url_model = None
    email_model = None
    nlp_pipeline = None
    logger.info("Running without ML models - using rule-based scoring")

# Initialize DNS resolver
dns_resolver = dns.resolver.Resolver()
dns_resolver.timeout = 2
dns_resolver.lifetime = 2

# Cache for WHOIS and DNS results
whois_cache = {}
dns_cache = {}

class URLFeatureExtractor:
    """Extracts advanced features from URLs for phishing detection"""
    
    URL_FEATURE_NAMES = [
        'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth', 'Redirection',
        'Non_HTTPS_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record',
        'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over', 'Right_Click',
        'Web_Forwards', 'SSL_Valid', 'Entropy', 'Brand_In_Domain',
        'Blacklisted', 'Redirect_Count', 'Domain_Popularity',
        'Similarity_Score', 'Anomaly_Score', 'Keyword_Density'
    ]
    
    def __init__(self):
        self.spell = SpellChecker()
        self.brands = list(brand_embeddings.keys()) if brand_embeddings else []
        self.tiny_url_pattern = re.compile(
            r"(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|buff\.ly|cutt\.ly|shorte\.st|bc\.vc|adf\.ly|u\.bb)"
        )
        self.script_pattern = re.compile(r'<script[^>]*>.*?</script>', re.DOTALL | re.IGNORECASE)
        self.phish_tank_api = "https://checkurl.phishtank.com/checkurl/"
        self.safe_browsing_api = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
        self.safe_browsing_key = os.environ.get('SAFE_BROWSING_API_KEY')
    
    def _calculate_similarity(self, url):
        """Calculate similarity to known phishing URLs using embeddings"""
        if not similarity_model or not brand_embeddings:
            return 0
            
        try:
            # Extract domain
            domain = tldextract.extract(url).domain.lower()
            
            # Skip if too short
            if len(domain) < 4:
                return 0
                
            domain_embedding = similarity_model.encode(domain)
            
            # Compare with brand embeddings
            max_similarity = 0
            for brand, embedding in brand_embeddings.items():
                similarity = cosine_similarity([domain_embedding], [embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
            
            # Compare with phishing corpus
            for phrase in phishing_corpus:
                phrase_embedding = similarity_model.encode(phrase)
                similarity = cosine_similarity([domain_embedding], [phrase_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
            
            # Convert to Python float before returning
            return float(min(max_similarity * 100, 100))
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {str(e)}")
            return 0.0

    def _calculate_keyword_density(self, url):
        """Calculate density of phishing keywords in URL"""
        if not phishing_keywords:
            return 0
            
        try:
            total_words = 0
            keyword_count = 0
            
            # Parse URL components
            parsed = urlparse(url)
            components = [parsed.netloc] + parsed.path.split('/') + parsed.query.split('&')
            
            for comp in components:
                words = re.findall(r'[a-zA-Z]{4,}', comp)
                total_words += len(words)
                for word in words:
                    if word.lower() in phishing_keywords:
                        keyword_count += 1
            
            if total_words == 0:
                return 0
                
            return min(keyword_count / total_words, 1)
        except Exception as e:
            logger.warning(f"Keyword density calculation failed: {str(e)}")
            return 0

    def _calculate_anomaly(self, features):
        """Calculate anomaly score using Isolation Forest"""
        if not anomaly_detector:
            return 0
            
        try:
            # Convert features to array (using only numerical features)
            feature_values = [
                features['Have_IP'],
                features['Have_At'],
                features['URL_Length'],
                features['URL_Depth'],
                features['Redirection'],
                features['Non_HTTPS_Domain'],
                features['TinyURL'],
                features['Prefix/Suffix'],
                features['Entropy'],
                features['Redirect_Count']
            ]
            score = anomaly_detector.decision_function([feature_values])[0]
            # Convert to Python float before returning
            return float((score + 0.5) * 100)
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {str(e)}")
            return 0.0

    def _having_ip(self, url):
        """Check if URL uses IP address instead of domain"""
        try:
            host = urlparse(url).netloc.split(':')[0]
            ipaddress.ip_address(host)
            return 1
        except (ValueError, IndexError):
            return 0

    def _have_at_sign(self, url):
        """Check for '@' in URL"""
        return 1 if "@" in url else 0

    def _get_length(self, url):
        """Check if URL exceeds safe length"""
        return 1 if len(url) >= 75 else 0

    def _get_depth(self, url):
        """Calculate path depth of URL"""
        path = urlparse(url).path
        return min(len([p for p in path.split('/') if p]), 10) / 10

    def _redirection(self, url):
        """Check for multiple redirections"""
        return 1 if url.rfind('//') > 7 else 0

    def _http_domain(self, url):
        """Check for insecure HTTP protocol"""
        return 0 if urlparse(url).scheme == 'https' else 1

    def _tiny_url(self, url):
        """Detect URL shortening services"""
        return 1 if self.tiny_url_pattern.search(url) else 0

    def _prefix_suffix(self, url):
        """Check for hyphens in domain"""
        return 1 if '-' in urlparse(url).netloc else 0

    def _domain_age(self, domain_name):
        """Check if domain is recently created"""
        try:
            if not domain_name:
                return 1
                
            # Check cache first
            if domain_name in whois_cache:
                return whois_cache[domain_name]['age']
                
            created = domain_name.creation_date
            if isinstance(created, list): 
                created = created[0]
            if isinstance(created, str):
                try:
                    created = datetime.strptime(created, '%Y-%m-%d')
                except ValueError:
                    return 1
                
            age_days = (datetime.now() - created).days
            result = 1 if age_days < 180 else 0
            
            # Cache result
            whois_cache[domain_name] = {
                'age': result,
                'expiry': None,
                'timestamp': datetime.now()
            }
            
            return result
        except Exception as e:
            logger.warning(f"Domain age error: {str(e)}")
            return 1

    def _domain_end(self, domain_name):
        """Check if domain expires soon"""
        try:
            if not domain_name:
                return 1
                
            # Check cache first
            if domain_name in whois_cache and whois_cache[domain_name]['expiry'] is not None:
                return whois_cache[domain_name]['expiry']
                
            expires = domain_name.expiration_date
            if isinstance(expires, list): 
                expires = expires[0]
            if isinstance(expires, str):
                try:
                    expires = datetime.strptime(expires, '%Y-%m-%d')
                except ValueError:
                    return 1
                
            days_left = (expires - datetime.now()).days
            result = 1 if days_left < 90 else 0
            
            # Update cache
            if domain_name in whois_cache:
                whois_cache[domain_name]['expiry'] = result
            else:
                whois_cache[domain_name] = {
                    'age': None,
                    'expiry': result,
                    'timestamp': datetime.now()
                }
                
            return result
        except Exception as e:
            logger.warning(f"Domain expiration error: {str(e)}")
            return 1

    def _iframe(self, response):
        """Detect hidden iframes"""
        if not response or not hasattr(response, 'text'):
            return 0
            
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            iframes = soup.find_all('iframe') + soup.find_all('frame')
            
            for iframe in iframes:
                # Check for hidden iframes
                style = iframe.get('style', '').lower()
                if 'display:none' in style or 'visibility:hidden' in style:
                    return 1
                    
                # Check for very small iframes
                width = iframe.get('width', '')
                height = iframe.get('height', '')
                if (width.isdigit() and int(width) < 5) or (height.isdigit() and int(height) < 5):
                    return 1
                    
            return 0
        except Exception:
            return 0

    def _mouse_over(self, response):
        """Check for mouseover events"""
        if not response or not hasattr(response, 'text'):
            return 0
        return 1 if re.search(r'onmouseover\s*=|onmouseenter\s*=', response.text, re.I) else 0

    def _right_click(self, response):
        """Check for disabled right-click"""
        if not response or not hasattr(response, 'text'):
            return 0
        return 1 if re.search(r'event\.button\s*==\s*2|contextmenu\s*=', response.text, re.I) else 0

    def _forwarding(self, response):
        """Check for excessive redirects"""
        if not response:
            return 0
        return min(len(response.history) / 5, 1)  # Normalize

    def _ssl_valid(self, url):
        """Check SSL certificate validity"""
        try:
            if urlparse(url).scheme != 'https':
                return 1
                
            # Check certificate validity
            hostname = urlparse(url).netloc
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
            # Check certificate expiration
            not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
            if (not_after - datetime.now()).days < 30:
                return 1
                
            return 0
        except (ssl.SSLError, socket.timeout, ConnectionRefusedError):
            return 1
        except Exception:
            return 1

    def _entropy(self, url):
        """Calculate Shannon entropy of URL"""
        text = urlparse(url).netloc
        if not text:
            return 0
            
        probs = [text.count(c) / len(text) for c in set(text)]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return min(entropy / 6, 1)  # Normalize

    def _brand_in_domain(self, url):
        """Check if legitimate brand is in subdomain"""
        domain = tldextract.extract(url).domain.lower()
        return 1 if any(brand in domain for brand in self.brands) else 0

    def _is_blacklisted(self, url):
        """Check URL against phishing databases"""
        try:
            # Create hash for caching
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            
            # Check cache
            if url_hash in dns_cache:
                return dns_cache[url_hash]
            
            result = 0
            
            # Check Google Safe Browsing
            if self.safe_browsing_key:
                payload = {
                    "client": {
                        "clientId": "phishguard",
                        "clientVersion": "1.0"
                    },
                    "threatInfo": {
                        "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
                        "platformTypes": ["ANY_PLATFORM"],
                        "threatEntryTypes": ["URL"],
                        "threatEntries": [{"url": url}]
                    }
                }
                response = requests.post(
                    f"{self.safe_browsing_api}?key={self.safe_browsing_key}",
                    json=payload,
                    timeout=3
                )
                if response.status_code == 200 and response.json().get('matches'):
                    result = 1
            
            # Check PhishTank if not already blacklisted
            if result == 0:
                response = requests.post(
                    self.phish_tank_api,
                    data={"url": url, "format": "json"},
                    headers={"User-Agent": "PhishGuard/4.0"},
                    timeout=3
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results', {}).get('in_database', False):
                        result = 1
            
            # Cache result
            dns_cache[url_hash] = result
            return result
        except Exception as e:
            logger.warning(f"Blacklist check failed: {str(e)}")
            return 0

    def _domain_popularity(self, domain):
        """Check domain popularity using DNS records"""
        try:
            # Check cache first
            if domain in dns_cache:
                return dns_cache[domain]
                
            # Check for common DNS records
            records_found = 0
            
            # Check MX records (email)
            try:
                dns_resolver.resolve(domain, 'MX')
                records_found += 1
            except: pass
            
            # Check TXT records (SPF, DMARC, etc.)
            try:
                dns_resolver.resolve(domain, 'TXT')
                records_found += 1
            except: pass
            
            # Check NS records
            try:
                dns_resolver.resolve(domain, 'NS')
                records_found += 1
            except: pass
            
            # More than one record indicates legitimate domain
            result = 1 if records_found < 2 else 0
            
            # Cache result
            dns_cache[domain] = result
            return result
        except Exception:
            return 0

    def _fetch_response(self, url):
        """Safely fetch URL content with error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
            
            # Capture redirect count
            redirect_count = len(response.history)
            
            return response, redirect_count
        except Exception as e:
            logger.warning(f"URL fetch failed: {str(e)}")
            return None, 0

    def extract_features(self, url):
        """Extract features from a single URL"""
        try:
            # Validate and clean URL
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
                
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ValueError("Invalid URL format")
                
            # Get WHOIS data
            try:
                domain_info = whois.whois(parsed.netloc)
                dns_record = 0
            except Exception:
                domain_info = None
                dns_record = 1
            
            # Fetch page content
            response, redirect_count = self._fetch_response(url)
            
            # Extract features
            url_features = {
                'Have_IP': self._having_ip(url),
                'Have_At': self._have_at_sign(url),
                'URL_Length': self._get_length(url),
                'URL_Depth': self._get_depth(url),
                'Redirection': self._redirection(url),
                'Non_HTTPS_Domain': self._http_domain(url),
                'TinyURL': self._tiny_url(url),
                'Prefix/Suffix': self._prefix_suffix(url),
                'DNS_Record': dns_record,
                'Domain_Age': self._domain_age(domain_info),
                'Domain_End': self._domain_end(domain_info),
                'iFrame': self._iframe(response),
                'Mouse_Over': self._mouse_over(response),
                'Right_Click': self._right_click(response),
                'Web_Forwards': self._forwarding(response),
                'SSL_Valid': self._ssl_valid(url),
                'Entropy': self._entropy(url),
                'Brand_In_Domain': self._brand_in_domain(url),
                'Blacklisted': self._is_blacklisted(url),
                'Redirect_Count': min(redirect_count / 5, 1),  # Normalize
                'Domain_Popularity': self._domain_popularity(parsed.netloc),
                'Similarity_Score': self._calculate_similarity(url),
                'Keyword_Density': self._calculate_keyword_density(url)
            }
            
            # Calculate anomaly score after other features
            url_features['Anomaly_Score'] = self._calculate_anomaly(url_features)
            
            return url_features
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            # Return default feature vector on error
            return {feature: 1 for feature in self.URL_FEATURE_NAMES}


class EmailFeatureExtractor:
    """Extracts advanced features from emails for phishing detection"""
    
    EMAIL_FEATURE_NAMES = [
        'Suspicious_Subject', 'Has_Links', 'Link_Count', 'Is_HTML', 'Has_Attachment',
        'Suspicious_Attachment', 'Sender_Mismatch', 'Auth_Fail', 'Shortened_URL',
        'Has_JS', 'Asks_Credentials', 'Excessive_CAPS', 'Generic_Greeting',
        'Threat_Language', 'Spelling_Errors', 'Sender_Reputation', 'DKIM_Valid',
        'DMARC_Pass', 'Content_Type_Mismatch', 'NLP_Score',
        'Syntax_Anomalies', 'Header_Inconsistencies', 'Keyword_Density'
    ]
    
    def __init__(self):
        self.spell = SpellChecker()
        self.credential_keywords = phishing_keywords or ['password', 'ssn', 'credit card', 'login', 'verify', 'account']
        self.threat_phrases = ['account suspension', 'legal action', 'immediately', 'urgent', 'verify now', 'limited time', 'action required', 'security alert']
        self.tiny_url_pattern = re.compile(
            r"(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|t\.co|is\.gd|buff\.ly|cutt\.ly|shorte\.st|bc\.vc|adf\.ly|u\.bb)"
        )
        self.suspicious_domains = re.compile(r"(@gmail\.com|@yahoo\.com|@outlook\.com|@hotmail\.com)$")
        self.dmarc_cache = {}
        self.header_cache = {}
    
    def _calculate_keyword_density(self, text):
        """Calculate density of phishing keywords in email content"""
        if not phishing_keywords:
            return 0
            
        try:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            if not words:
                return 0
                
            keyword_count = sum(1 for word in words if word in phishing_keywords)
            return min(keyword_count / len(words), 1)
        except Exception as e:
            logger.warning(f"Keyword density calculation failed: {str(e)}")
            return 0

    def _detect_syntax_anomalies(self, email_features):
        """Detect email syntax anomalies using ML"""
        try:
            # Extract features for anomaly detection
            features = [
                email_features['Suspicious_Subject'],
                email_features['Link_Count'],
                email_features['Has_Attachment'],
                email_features['Asks_Credentials'],
                email_features['Spelling_Errors']
            ]
            
            if anomaly_detector:
                score = anomaly_detector.decision_function([features])[0]
                return float((score + 0.5) * 100)  # Convert to Python float
            return 0.0
        except Exception as e:
            logger.warning(f"Syntax anomaly detection failed: {str(e)}")
            return 0.0

    def _detect_header_inconsistencies(self, headers):
        """Detect email header inconsistencies"""
        try:
            # Create hash for caching
            header_hash = hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest()
            
            if header_hash in self.header_cache:
                return self.header_cache[header_hash]
                
            inconsistencies = 0
            
            # Check date format
            if 'Date' in headers:
                try:
                    datetime.strptime(headers['Date'], '%a, %d %b %Y %H:%M:%S %z')
                except ValueError:
                    inconsistencies += 1
            
            # Check for missing headers
            for header in ['From', 'To', 'Subject', 'Date']:
                if header not in headers:
                    inconsistencies += 1
            
            # Check for multiple Received headers with different domains
            received_headers = [h for h in headers.keys() if h.lower() == 'received']
            if len(received_headers) > 1:
                domains = set()
                for header in received_headers:
                    if 'from ' in headers[header].lower():
                        domain = headers[header].split('from ')[-1].split()[0]
                        domains.add(domain)
                if len(domains) > 2:
                    inconsistencies += 1
            
            # Check sender/receiver consistency
            if 'From' in headers and 'To' in headers:
                from_domain = headers['From'].split('@')[-1] if '@' in headers['From'] else ''
                to_domain = headers['To'].split('@')[-1] if '@' in headers['To'] else ''
                if from_domain and to_domain and from_domain == to_domain:
                    inconsistencies += 1
            
            result = min(inconsistencies / 5, 1)  # Normalize to 0-1
            
            # Cache result
            self.header_cache[header_hash] = result
            return result
        except Exception as e:
            logger.warning(f"Header inconsistency detection failed: {str(e)}")
            return 0

    def _get_nlp_score(self, text):
        """Get NLP sentiment score for email content"""
        if not nlp_pipeline or not text:
            return 0.5
            
        try:
            result = nlp_pipeline(text[:512])  # Limit to first 512 characters
            # Convert sentiment to score: POSITIVE=1, NEGATIVE=0
            return 0 if result[0]['label'] == 'NEGATIVE' else 1
        except Exception:
            return 0.5

    def _suspicious_subject(self, subject):
        """Detect urgency/scare tactics in subject"""
        keywords = ['urgent', 'action required', 'important', 'verify', 'security alert', 'account']
        return 1 if any(k in subject.lower() for k in keywords) else 0

    def _has_links(self, body):
        """Check for links in email body"""
        return 1 if re.search(r'http[s]?://|www\.', body) else 0

    def _count_links(self, body):
        """Count number of links in body"""
        return min(len(re.findall(r'http[s]?://[^\s<>"]+|www\.[^\s<>"]+', body)) / 5, 1)

    def _is_html(self, body):
        """Check if email contains HTML"""
        return 1 if re.search(r'<[^>]+>', body) else 0

    def _has_attachments(self, attachments):
        """Check for attachments"""
        return 1 if attachments else 0

    def _suspicious_attachment(self, attachments):
        """Detect dangerous attachment types"""
        suspicious_ext = ('.exe', '.bat', '.scr', '.js', '.vbs', '.jar', '.cmd', '.msi', '.dll', '.zip', '.rar')
        for att in attachments:
            if any(att.lower().endswith(ext) for ext in suspicious_ext):
                return 1
        return 0

    def _sender_mismatch(self, headers):
        """Check for spoofed sender"""
        from_header = headers.get('From', '').lower()
        return_path = headers.get('Return-Path', '').lower()
        return 1 if from_header and return_path and from_header != return_path else 0

    def _auth_fail(self, headers):
        """Check authentication failures"""
        spf = headers.get('Received-SPF', '').lower()
        auth = headers.get('Authentication-Results', '').lower()
        return 1 if 'fail' in spf or 'fail' in auth else 0

    def _shortened_url_in_body(self, body):
        """Detect URL shortening services"""
        return 1 if self.tiny_url_pattern.search(body) else 0

    def _has_javascript(self, body):
        """Check for JavaScript in HTML"""
        return 1 if '<script' in body.lower() else 0

    def _asks_for_credentials(self, body):
        """Detect credential requests"""
        return 1 if any(k in body.lower() for k in self.credential_keywords) else 0

    def _excessive_caps(self, subject):
        """Detect ALL-CAPS subject lines"""
        if not subject:
            return 0
        upper_count = sum(1 for char in subject if char.isupper())
        return 1 if upper_count / len(subject) > 0.6 else 0

    def _generic_greeting(self, body):
        """Detect impersonal greetings"""
        greetings = ['dear user', 'dear customer', 'valued member', 'dear account holder', 'dear client']
        return 1 if any(g in body.lower() for g in greetings) else 0

    def _has_threats(self, body):
        """Detect threatening language"""
        return 1 if any(t in body.lower() for t in self.threat_phrases) else 0

    def _spelling_errors(self, body):
        """Count spelling mistakes in visible text"""
        try:
            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', '', body)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', clean_text)
            unknown_words = self.spell.unknown(words)
            return min(len(unknown_words) / len(words) if words else 0, 1)  # Normalize
        except Exception:
            return 0

    def _sender_reputation(self, headers):
        """Check sender domain reputation"""
        from_domain = headers.get('From', '').split('@')[-1] if '@' in headers.get('From', '') else ''
        return 1 if from_domain and self.suspicious_domains.search(from_domain) else 0

    def _dkim_valid(self, headers):
        """Check DKIM validation"""
        dkim = headers.get('DKIM-Signature', '')
        auth = headers.get('Authentication-Results', '')
        return 0 if 'dkim=pass' in auth.lower() and dkim else 1

    def _dmarc_pass(self, headers):
        """Check DMARC validation"""
        dmarc = headers.get('DMARC-Results', '')
        auth = headers.get('Authentication-Results', '')
        
        # If not in headers, check DNS
        if 'dmarc=pass' in auth.lower() and dmarc:
            return 0
            
        # Check DNS DMARC record
        domain = headers.get('From', '').split('@')[-1] if '@' in headers.get('From', '') else ''
        if not domain:
            return 1
            
        # Check cache first
        if domain in self.dmarc_cache:
            return 1 if self.dmarc_cache[domain] == 'fail' else 0
            
        try:
            answers = dns_resolver.resolve(f'_dmarc.{domain}', 'TXT')
            for rdata in answers:
                if 'v=DMARC1' in rdata.strings[0]:
                    self.dmarc_cache[domain] = 'pass'
                    return 0
            self.dmarc_cache[domain] = 'fail'
            return 1
        except:
            self.dmarc_cache[domain] = 'fail'
            return 1

    def _content_type_mismatch(self, headers):
        """Check for content type inconsistencies"""
        ctype = headers.get('Content-Type', '')
        if 'multipart/mixed' in ctype and 'text/html' not in ctype:
            return 1
        return 0

    def extract_features(self, email):
        """Extract features from email object"""
        try:
            subject = email.get("subject", "")
            body = email.get("body", "")
            headers = email.get("headers", {})
            attachments = email.get("attachments", [])
            
            # Extract NLP sentiment score
            nlp_score = self._get_nlp_score(subject + " " + body)
            
            # Extract features
            email_features = {
                'Suspicious_Subject': self._suspicious_subject(subject),
                'Has_Links': self._has_links(body),
                'Link_Count': self._count_links(body),
                'Is_HTML': self._is_html(body),
                'Has_Attachment': self._has_attachments(attachments),
                'Suspicious_Attachment': self._suspicious_attachment(attachments),
                'Sender_Mismatch': self._sender_mismatch(headers),
                'Auth_Fail': self._auth_fail(headers),
                'Shortened_URL': self._shortened_url_in_body(body),
                'Has_JS': self._has_javascript(body),
                'Asks_Credentials': self._asks_for_credentials(body),
                'Excessive_CAPS': self._excessive_caps(subject),
                'Generic_Greeting': self._generic_greeting(body),
                'Threat_Language': self._has_threats(body),
                'Spelling_Errors': self._spelling_errors(body),
                'Sender_Reputation': self._sender_reputation(headers),
                'DKIM_Valid': self._dkim_valid(headers),
                'DMARC_Pass': self._dmarc_pass(headers),
                'Content_Type_Mismatch': self._content_type_mismatch(headers),
                'NLP_Score': nlp_score,
                'Keyword_Density': self._calculate_keyword_density(body),
                'Header_Inconsistencies': self._detect_header_inconsistencies(headers)
            }
            
            # Add AI-based features
            email_features['Syntax_Anomalies'] = self._detect_syntax_anomalies(email_features)
            
            return email_features
            
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
            # Return default feature vector on error
            return {feature: 1 for feature in self.EMAIL_FEATURE_NAMES}


# Forms
class URLForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired()])

class EmailForm(FlaskForm):
    email = TextAreaField('Email Content', validators=[DataRequired()])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    # Initialize session if needed
    if 'scan_history' not in session:
        session['scan_history'] = []
    
    # Load scan history from session
    scan_history = session.get('scan_history', [])
    
    # Prepare statistics
    url_scans = [s for s in scan_history if s['type'] == 'url']
    email_scans = [s for s in scan_history if s['type'] == 'email']
    
    stats = {
        'total_scans': len(scan_history),
        'url_scans': len(url_scans),
        'email_scans': len(email_scans),
        'high_risk': len([s for s in scan_history if s['risk'] > 70]),
        'medium_risk': len([s for s in scan_history if s['risk'] > 30 and s['risk'] <= 70]),
        'low_risk': len([s for s in scan_history if s['risk'] <= 30])
    }
    
    return render_template('dashboard.html', stats=stats, history=scan_history[:10])


@app.route('/get-csrf-token', methods=['GET'])
def get_csrf_token():
    return jsonify({'csrf_token': generate_csrf()})


@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    # Extract features
    extractor = URLFeatureExtractor()
    features = extractor.extract_features(url)
    
    # Calculate risk score
    risk_score = calculate_risk_score(features)
    
    # Prepare features for frontend
    display_features = [
        {'name': 'IP Address', 'value': features['Have_IP'], 'icon': 'fa-globe'},
        {'name': '@ Symbol', 'value': features['Have_At'], 'icon': 'fa-at'},
        {'name': 'Long URL', 'value': features['URL_Length'], 'icon': 'fa-ruler'},
        {'name': 'Redirection', 'value': features['Redirection'], 'icon': 'fa-directions'},
        {'name': 'Non-HTTPS', 'value': features['Non_HTTPS_Domain'], 'icon': 'fa-lock-open'},
        {'name': 'TinyURL', 'value': features['TinyURL'], 'icon': 'fa-compress'},
        {'name': 'Prefix/Suffix', 'value': features['Prefix/Suffix'], 'icon': 'fa-minus'},
        {'name': 'DNS Issues', 'value': features['DNS_Record'], 'icon': 'fa-server'},
        {'name': 'Blacklisted', 'value': features['Blacklisted'], 'icon': 'fa-ban'},
        {'name': 'Similarity', 'value': features['Similarity_Score']/100, 'icon': 'fa-percentage'},
        {'name': 'Anomaly', 'value': features['Anomaly_Score']/100, 'icon': 'fa-chart-line'}
    ]
    
    # Prepare radar chart data
    radar_data = [
        features['Have_IP'], features['Have_At'], features['URL_Length'],
        features['Redirection'], features['Non_HTTPS_Domain'], 
        features['TinyURL'], features['Prefix/Suffix'], features['DNS_Record'],
        features['Blacklisted'], features['Similarity_Score']/100
    ]
    
    # Add to session history
    scan_history = session.get('scan_history', [])
    scan_history.insert(0, {
        'type': 'url',
        'content': url,
        'risk': risk_score,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': features
    })
    session['scan_history'] = scan_history[:50]  # Keep last 50 scans
    
    return jsonify({
        'risk_score': risk_score,
        'features': display_features,
        'radar_data': radar_data,
        'url': url,
        'is_blacklisted': bool(features['Blacklisted']),
        'similarity_score': features['Similarity_Score'],
        'anomaly_score': features['Anomaly_Score']
    })


@app.route('/analyze-email', methods=['POST'])
def analyze_email():
    data = request.get_json()
    email_content = data.get('email', '')
    
    if not email_content:
        return jsonify({'error': 'Email content is required'}), 400
    
    # Create email object
    email_obj = {
        "subject": "",
        "body": email_content,
        "headers": {},
        "attachments": []
    }
    
    # Extract features
    extractor = EmailFeatureExtractor()
    features = extractor.extract_features(email_obj)
    
    # Calculate risk score
    risk_score = calculate_email_risk_score(features)
    
    # Prepare features for frontend
    display_features = [
        {'name': 'Suspicious Subject', 'value': features['Suspicious_Subject'], 'icon': 'fa-heading'},
        {'name': 'Contains Links', 'value': features['Has_Links'], 'icon': 'fa-link'},
        {'name': 'HTML Content', 'value': features['Is_HTML'], 'icon': 'fa-code'},
        {'name': 'Attachments', 'value': features['Has_Attachment'], 'icon': 'fa-paperclip'},
        {'name': 'Sender Mismatch', 'value': features['Sender_Mismatch'], 'icon': 'fa-user'},
        {'name': 'Auth Failure', 'value': features['Auth_Fail'], 'icon': 'fa-shield-alt'},
        {'name': 'Credential Request', 'value': features['Asks_Credentials'], 'icon': 'fa-key'},
        {'name': 'Threat Language', 'value': features['Threat_Language'], 'icon': 'fa-exclamation-triangle'},
        {'name': 'Spelling Errors', 'value': features['Spelling_Errors'], 'icon': 'fa-spell-check'},
        {'name': 'NLP Score', 'value': features['NLP_Score'], 'icon': 'fa-brain'},
        {'name': 'Syntax Anomalies', 'value': features['Syntax_Anomalies']/100, 'icon': 'fa-bug'},
        {'name': 'Header Issues', 'value': features['Header_Inconsistencies'], 'icon': 'fa-envelope'}
    ]
    
    # Prepare radar chart data
    radar_data = [
        features['Suspicious_Subject'], features['Has_Links'], 
        features['Is_HTML'], features['Has_Attachment'], 
        features['Sender_Mismatch'], features['Auth_Fail'], 
        features['Asks_Credentials'], features['Threat_Language'],
        features['NLP_Score'], features['Syntax_Anomalies']/100
    ]
    
    # Add to session history
    scan_history = session.get('scan_history', [])
    scan_history.insert(0, {
        'type': 'email',
        'content': email_content[:100] + '...' if len(email_content) > 100 else email_content,
        'risk': risk_score,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': features
    })
    session['scan_history'] = scan_history[:50]  # Keep last 50 scans
    
    return jsonify({
        'risk_score': risk_score,
        'features': display_features,
        'radar_data': radar_data,
        'email_preview': email_content[:50] + '...' if len(email_content) > 50 else email_content,
        'nlp_score': features['NLP_Score'] * 100,
        'syntax_anomalies': features['Syntax_Anomalies']
    })


@app.route('/export-history')
def export_history():
    scan_history = session.get('scan_history', [])
    if not scan_history:
        return jsonify({'error': 'No scan history available'}), 404
    
    # Convert to CSV
    csv_data = "Type,Content,Risk,Date\n"
    for item in scan_history:
        content = item['content'].replace(',', ';')  # Sanitize for CSV
        csv_data += f"{item['type']},{content},{item['risk']},{item['date']}\n"
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=phishguard_history.csv"}
    )


@app.route('/report', methods=['POST'])
def generate_report():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Generate report ID
    report_id = str(uuid.uuid4())[:8].upper()
    
    # Generate PDF report (simplified for example)
    report = f"""
    PhishGuard Analysis Report
    =========================
    
    Report ID: {report_id}
    Scan Type: {data.get('type', 'Unknown').upper()}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Risk Score: {data.get('risk_score', 0)}%
    
    Content:
    {data.get('content', '')[:500] + '...' if len(data.get('content', '')) > 500 else data.get('content', '')}
    
    Features:
    """
    
    for feature in data.get('features', []):
        report += f"\n- {feature['name']}: {'Yes' if feature['value'] else 'No'}"
    
    # Add AI insights
    if data.get('similarity_score'):
        report += f"\n\nAI Insights:\n- Brand Similarity: {data['similarity_score']:.1f}%"
    if data.get('anomaly_score'):
        report += f"\n- Anomaly Detection: {data['anomaly_score']:.1f}%"
    if data.get('syntax_anomalies'):
        report += f"\n- Syntax Anomalies: {data['syntax_anomalies']:.1f}%"
    
    report += "\n\nConclusion: " + (
        "HIGH RISK - Likely phishing attempt" if data.get('risk_score', 0) > 70 else
        "MEDIUM RISK - Suspicious characteristics detected" if data.get('risk_score', 0) > 30 else
        "LOW RISK - Appears legitimate"
    )
    
    return jsonify({
        'report': report,
        'report_id': report_id,
        'message': 'PDF report generated successfully'
    })


def calculate_risk_score(features):
    """Calculate phishing risk score with ML model if available"""
    if url_model:
        try:
            # Prepare features in correct order
            feature_values = [features[feature] for feature in URLFeatureExtractor.URL_FEATURE_NAMES]
            # Predict using ML model
            risk_score = url_model.predict_proba([feature_values])[0][1] * 100
            return min(100, int(risk_score))
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
    
    # Fallback to rule-based scoring
    weights = {
        'Have_IP': 0.12, 'Have_At': 0.08, 'URL_Length': 0.05, 
        'Redirection': 0.09, 'Non_HTTPS_Domain': 0.12, 'TinyURL': 0.08, 
        'Prefix/Suffix': 0.04, 'DNS_Record': 0.05, 'iFrame': 0.07, 
        'Brand_In_Domain': 0.10, 'Blacklisted': 0.15, 'Redirect_Count': 0.05,
        'Similarity_Score': 0.20, 'Anomaly_Score': 0.10
    }
    
    score = 0
    for feature, weight in weights.items():
        # Normalize scores that are percentages
        if feature in ['Similarity_Score', 'Anomaly_Score']:
            score += (features.get(feature, 0) / 100) * weight
        else:
            score += features.get(feature, 0) * weight
    
    return min(100, int(score * 100))


def calculate_email_risk_score(features):
    """Calculate phishing risk score with ML model if available"""
    if email_model:
        try:
            # Prepare features in correct order
            feature_values = [features[feature] for feature in EmailFeatureExtractor.EMAIL_FEATURE_NAMES]
            # Predict using ML model
            risk_score = email_model.predict_proba([feature_values])[0][1] * 100
            return min(100, int(risk_score))
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
    
    # Fallback to rule-based scoring
    weights = {
        'Suspicious_Subject': 0.09,
        'Has_Links': 0.09,
        'Is_HTML': 0.04,
        'Has_Attachment': 0.04,
        'Suspicious_Attachment': 0.12,
        'Sender_Mismatch': 0.12,
        'Auth_Fail': 0.08,
        'Shortened_URL': 0.08,
        'Asks_Credentials': 0.15,
        'NLP_Score': 0.10,
        'Syntax_Anomalies': 0.05,
        'Header_Inconsistencies': 0.04
    }
    
    score = 0
    for feature, weight in weights.items():
        # Normalize scores that are percentages
        if feature in ['Syntax_Anomalies']:
            score += (features.get(feature, 0) / 100) * weight
        else:
            score += features.get(feature, 0) * weight
    
    return min(100, int(score * 100))


# Clean up cache periodically
def cleanup_cache():
    now = datetime.now()
    # Clean WHOIS cache
    for domain in list(whois_cache.keys()):
        if now - whois_cache[domain]['timestamp'] > timedelta(hours=24):
            del whois_cache[domain]
    
    # Clean DNS cache
    for key in list(dns_cache.keys()):
        if now - dns_cache[key]['timestamp'] > timedelta(minutes=60):
            del dns_cache[key]


if __name__ == "__main__":
    # Start periodic cache cleanup
    def cache_cleaner():
        while True:
            cleanup_cache()
            time.sleep(3600)  # Clean every hour
    
    cleaner_thread = threading.Thread(target=cache_cleaner, daemon=True)
    cleaner_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)