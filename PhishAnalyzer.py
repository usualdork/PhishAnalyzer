import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
import logging
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('phishanalyzer.log'),
            logging.StreamHandler()
        ]
    )
    logging.debug("Logging setup completed")

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    logging.debug("NLTK downloads completed successfully")
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")

class DatasetHandler:
    """Handle dataset loading and processing"""
    
    def __init__(self):
        self.data_dir = "datasets"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        logging.debug("DatasetHandler initialized")
    
    def extract_url_features(self, df):
        """Extract features from URLs"""
        logging.debug("Extracting URL features...")
        
        def get_url_features(url):
            try:
                parsed = urlparse(url)
                
                # Basic URL features
                return {
                    'length': len(url),
                    'has_ip': self._has_ip(url),
                    'has_at': '@' in url,
                    'has_double_slash': '//' in parsed.path,
                    'subdomain_count': len(parsed.netloc.split('.')) - 1,
                    'is_https': parsed.scheme == 'https',
                    'has_suspicious_words': self._has_suspicious_words(url),
                    'domain_length': len(parsed.netloc),
                    'path_length': len(parsed.path),
                    'query_length': len(parsed.query),
                    'fragment_length': len(parsed.fragment),
                    'num_digits': sum(c.isdigit() for c in url),
                    'num_params': len(parsed.query.split('&')) if parsed.query else 0
                }
            except Exception as e:
                logging.error(f"Error extracting features from URL {url}: {str(e)}")
                # Return default features if extraction fails
                return {
                    'length': 0, 'has_ip': False, 'has_at': False,
                    'has_double_slash': False, 'subdomain_count': 0,
                    'is_https': False, 'has_suspicious_words': False,
                    'domain_length': 0, 'path_length': 0, 'query_length': 0,
                    'fragment_length': 0, 'num_digits': 0, 'num_params': 0
                }

        # Extract features for each URL
        features_list = []
        for url in df['url']:
            features = get_url_features(url)
            features_list.append(features)
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add the spam label
        features_df['is_spam'] = df['is_spam']
        
        logging.debug(f"URL features extracted. Shape: {features_df.shape}")
        return features_df
    
    def _has_ip(self, url):
        """Check if URL contains an IP address"""
        pattern = r'(?:\d{1,3}\.){3}\d{1,3}'
        return bool(re.search(pattern, url))
    
    def _has_suspicious_words(self, url):
        """Check for suspicious words in URL"""
        suspicious = [
            'login', 'verify', 'account', 'secure', 'banking', 'update',
            'confirm', 'password', 'verify', 'authentication', 'authorize',
            'wallet', 'credential', 'signin', 'security', 'payment'
        ]
        url_lower = url.lower()
        return any(word in url_lower for word in suspicious)
            
    def load_email_dataset(self, filepath):
        """Load and process the email dataset"""
        logging.info(f"Loading Email Dataset from {filepath}...")
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Email dataset file not found: {filepath}")
            
            # Load the CSV file
            df = pd.read_csv(filepath)
            logging.debug(f"Email dataset shape: {df.shape}")
            
            # Ensure column names are correct
            df.columns = ['label', 'text']
            
            # Convert labels to numeric if they aren't already
            df['label'] = df['label'].astype(int)
            
            logging.info("Successfully loaded email dataset")
            return df
            
        except Exception as e:
            logging.error(f"Error loading email dataset: {str(e)}")
            raise

    def load_url_dataset(self, filepath):
        """Load and process the URL dataset"""
        logging.info(f"Loading URL Dataset from {filepath}...")
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"URL dataset file not found: {filepath}")
            
            # Load the CSV file
            df = pd.read_csv(filepath)
            logging.debug(f"URL dataset shape: {df.shape}")
            
            # Ensure column names are correct
            df.columns = ['url', 'is_spam']
            
            # Convert boolean/string labels to numeric
            df['is_spam'] = df['is_spam'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
            
            # Extract URL features
            df = self.extract_url_features(df)
            
            logging.info("Successfully loaded URL dataset")
            return df
            
        except Exception as e:
            logging.error(f"Error loading URL dataset: {str(e)}")
            raise



class PhishingDetector:
    """Main class for phishing detection with added validation metrics"""
    
    def __init__(self):
        self.email_vectorizer = TfidfVectorizer(max_features=1000)
        self.url_classifier = RandomForestClassifier(n_estimators=100)
        self.email_classifier = MultinomialNB()
        self.metrics = {}
        logging.debug("PhishingDetector initialized")
    
    def preprocess_email(self, email):
        """Preprocess email content"""
        # Convert to lowercase
        email = str(email).lower()
        
        # Remove special characters
        email = re.sub(r'[^a-zA-Z\s]', ' ', email)
        
        # Tokenization
        tokens = word_tokenize(email)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def evaluate_model(self, X, y, model, model_name):
        """Evaluate model using cross-validation and multiple metrics"""
        metrics = {}
        
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        metrics['cv_accuracy'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Split data for detailed metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model on split
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate various metrics
        metrics['test_accuracy'] = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['precision'] = class_report['weighted avg']['precision']
        metrics['recall'] = class_report['weighted avg']['recall']
        metrics['f1_score'] = class_report['weighted avg']['f1-score']
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = conf_matrix
        
        # Log results
        logging.info(f"\nMetrics for {model_name}:")
        logging.info(f"Cross-validation accuracy: {metrics['cv_accuracy']:.3f} (±{metrics['cv_std']:.3f})")
        logging.info(f"Test accuracy: {metrics['test_accuracy']:.3f}")
        logging.info(f"Precision: {metrics['precision']:.3f}")
        logging.info(f"Recall: {metrics['recall']:.3f}")
        logging.info(f"F1 Score: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def train(self, email_data, url_data):
        """Train both email and URL classifiers with comprehensive validation"""
        logging.info("Starting model training and validation...")
        
        # Process email data
        X_email = [self.preprocess_email(email) for email in email_data['text']]
        X_email = self.email_vectorizer.fit_transform(X_email)
        y_email = email_data['label']
        
        # Process URL data
        X_url = url_data.drop('is_spam', axis=1)
        y_url = url_data['is_spam']
        
        # Evaluate and train email classifier
        self.metrics['email'] = self.evaluate_model(
            X_email, y_email, 
            self.email_classifier, 
            "Email Classifier"
        )
        
        # Evaluate and train URL classifier
        self.metrics['url'] = self.evaluate_model(
            X_url, y_url,
            self.url_classifier,
            "URL Classifier"
        )
        
        # Final training on full dataset
        self.email_classifier.fit(X_email, y_email)
        self.url_classifier.fit(X_url, y_url)
        
        logging.info("Model training and validation completed")
        return self.metrics
    
    def predict(self, email_content):
        """Predict if an email is phishing with confidence metrics"""
        # Preprocess and vectorize email
        processed_email = self.preprocess_email(email_content)
        email_features = self.email_vectorizer.transform([processed_email])
        
        # Get prediction probability
        email_prob = self.email_classifier.predict_proba(email_features)[0][1]
        
        # Extract URLs from email
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                         email_content)
        
        result = {
            'email_phishing_probability': email_prob,
            'urls_found': len(urls),
            'is_phishing': email_prob > 0.5,
            'risk_level': 'High' if email_prob > 0.7 else 'Medium' if email_prob > 0.3 else 'Low',
            'model_metrics': {
                'email_classifier_cv_accuracy': self.metrics.get('email', {}).get('cv_accuracy', None),
                'url_classifier_cv_accuracy': self.metrics.get('url', {}).get('cv_accuracy', None)
            }
        }
        
        return result



def main():
    """Main function to demonstrate the system's functionality with validation metrics"""
    try:
        # Setup logging first
        setup_logging()
        
        print("\n=== Starting PhishAnalyzer AI Security System ===")
        logging.info("Starting PhishAnalyzer AI Security System...")
        
        # Initialize DatasetHandler and load datasets  
        data_handler = DatasetHandler() # Initialize DatasetHandler object
        email_data = data_handler.load_email_dataset("/content/email_dataset.csv") # Load email dataset
        url_data = data_handler.load_url_dataset("/content/url_dataset.csv")    # Load URL dataset
        
        print("\nInitializing and training models with validation...")
        detector = PhishingDetector()
        metrics = detector.train(email_data, url_data)
        
        
        print("\n=== Model Performance Metrics ===")
        
        # Display Email Classifier Metrics
        print("\nEmail Classifier Metrics:")
        print(f"Cross-validation Accuracy: {metrics['email']['cv_accuracy']:.3f} (±{metrics['email']['cv_std']:.3f})")
        print(f"Test Accuracy: {metrics['email']['test_accuracy']:.3f}")
        print(f"Precision: {metrics['email']['precision']:.3f}")
        print(f"Recall: {metrics['email']['recall']:.3f}")
        print(f"F1 Score: {metrics['email']['f1_score']:.3f}")
        
        # Display URL Classifier Metrics
        print("\nURL Classifier Metrics:")
        print(f"Cross-validation Accuracy: {metrics['url']['cv_accuracy']:.3f} (±{metrics['url']['cv_std']:.3f})")
        print(f"Test Accuracy: {metrics['url']['test_accuracy']:.3f}")
        print(f"Precision: {metrics['url']['precision']:.3f}")
        print(f"Recall: {metrics['url']['recall']:.3f}")
        print(f"F1 Score: {metrics['url']['f1_score']:.3f}")
        
        # Test the system with example emails
        test_emails = [
            '''
            Subject: Prize Notification
            Microsoft Uberica S.L Lottery Intl. Program FOREIGN SERVICE SECTION BARCELONA. REFERENCE NUMBER:YUKFQ/RYYHJ
            BATCH NUMBER: 2016/WTN
            OFFICIAL WINNING NOTIFICATION.
            We are pleased to inform you Of the released results Of the Microsoft S.L Sweepstakes Promotion in conjunction
            with foundations for the promotion of software products organized for Software users. This Program was held in Barcelona-
            Spain; Wherein your email address emerged as one of the online Winning emails in the 1st category and therefore attracted
            a cash award of EUR344m)M and a Mac laptop/iPhone. Your laptop, certificate of winnings and your cheque of
            will be sent to your contact address in your location. TO file for claims Of the release Of your winnings,
            contact the Customer Service Officer with the information below:
            FULL NAMES, ADDRESS, SEX, AGE, MARITAL STATUS, OCCUPATION, TELEPHONE NUMBER, COUNTRY, BATCH NUMBER,
            REFERENCE NUMBER: Email: cuservdept@excite.co.ip Contact Person: Manuel Vizner (CSOI
            Also, please, fill out our customer satisfaction survey at www.excite.co.ip/Survey.aspx?s=4674c60&surv id=DNTY
            Congratulations!!
            Sincerely,
            Mrs. Miriam
            Online Coordinator
            ''',
            '''
            Attn:
            Board Members And Directors Agreed Today That your over due
            payment/lnheritance/Contract Fund valued at $3.7 Million Will be Released
            to you On A Special Method Payment.via ATM master debit card OR key telex
            transfer (KTT) direct wire transfer, You Are to contact with your
            information immediately. Full name,Address,Phone,age,occupation to claim
            your funds.
            Waiting to hear from you soon, you can call me on Tel-
            +234-807-158-0925 for more details.
            Thanks
            Chris Daniel
            Tel- +234-807-158-0925
            ''',
            '''
            Hello, thank you for your questions.

            Internships can be paid. Since we can sponsor internships in a variety of fields, the kinds of monetary and non-monetary compensation varies widely depending on the intern and their company. However, if the internship will exceed 6 months, the host company must pay the intern at least the minimum wage or higher. Ultimately, it will be up to you and your host to discuss what monetary and/or non-monetary compensation the host company will provide.
            Shelby

            StuentAffairs
            '''
        ]
        
        print("\n=== Testing PhishAnalyzer with Example Emails ===")
        
        for i, email in enumerate(test_emails, 1):
            result = detector.predict(email)
            print(f"\nResults for Test Email {i}:")
            print(f"Phishing Probability: {result['email_phishing_probability']:.2f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"URLs Found: {result['urls_found']}")
            print(f"Is Phishing: {result['is_phishing']}")
        
        print("\nPhishAnalyzer testing completed successfully!")
        
    except Exception as e:
        print(f"\nERROR during execution: {str(e)}")
        logging.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
