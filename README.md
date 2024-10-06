# PhishAnalyzer
PhishAnalyzer is a machine learning-based security tool that combines email content analysis and URL feature extraction to detect phishing attempts. It employs a dual-layer approach using Natural Language Processing (NLP) and Random Forest classification to provide comprehensive phishing detection with detailed risk assessments.

## Key Features

* Dual-Layer Analysis: Combines email content and URL feature analysis
* Advanced URL Feature Extraction: Analyzes 13 distinct URL characteristics
* Comprehensive Metrics: Includes cross-validation, precision, recall, and F1 scores
* Risk Assessment: Provides probability scores and risk levels (High/Medium/Low)
* Detailed Logging: Maintains comprehensive logs for analysis and debugging

## Technical Architecture

* Email Classification: Multinomial Naive Bayes with TF-IDF vectorization
* URL Classification: Random Forest with engineered features
* Text Processing: NLTK for tokenization and stopword removal
* Validation: 5-fold cross-validation with multiple performance metrics

## Installation
```
#Clone the repository
git clone https://github.com/your-username/phishanalyzer.git
cd phishanalyzer

#Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

#Install requirements
pip install -r requirements.txt

#Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage
1. Prepare your datasets in CSV format:
   * Email dataset: columns=['label', 'text']
   * URL dataset: columns=['url', 'is_spam']
2. Run the analyzer:
```python phishanalyzer.py```

## Example Output
```
=== Model Performance Metrics ===

Email Classifier Metrics:
Cross-validation Accuracy: 0.945 (±0.015)
Test Accuracy: 0.952
Precision: 0.953
Recall: 0.952
F1 Score: 0.952

URL Classifier Metrics:
Cross-validation Accuracy: 0.938 (±0.012)
Test Accuracy: 0.941
Precision: 0.942
Recall: 0.941
F1 Score: 0.941
```

## Dataset Credits
This project uses the following datasets from Kaggle:

1. Email Spam Classification Dataset
   * Source: Kaggle - Email Spam Classification Dataset CSV
   *  Author: Balaka Biswas
2. URL Spam Prediction Dataset
   * Source: Kaggle - Spam URL Prediction
   *  Author: Shivam Bansal

## Development Credits
This project was developed with assistance from:
* Anthropic's Claude AI for code enhancement and optimization
* Various open-source libraries (see requirements.txt)

## Requirements
See ```requirements.txt``` for a complete list of dependencies. Key requirements:

* Python 3.8+
* scikit-learn
* pandas
* numpy
* nltk

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer
This tool is for educational and research purposes only. While it can help identify potential phishing attempts, it should not be the sole mechanism for email security.
