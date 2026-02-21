import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

class ResumePreprocessor:
    """
    Cleans and preprocesses resume text for ML model
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Complete text cleaning pipeline
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Step 3: Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Step 4: Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'\+\d{1,3}\s?\d+', '', text)
        
        # Step 5: Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Step 6: Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize the text
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        # Tokenize and lemmatize
        processed = self.tokenize_and_lemmatize(cleaned)
        
        return processed

# Test the preprocessor
if __name__ == "__main__":
    preprocessor = ResumePreprocessor()
    
    sample_text = """
    John Doe
    Email: john@example.com
    Phone: +1-555-123-4567
    
    SKILLS:
    Python, Machine Learning, Data Analysis, SQL, TensorFlow
    Experience with pandas, numpy, scikit-learn
    
    EXPERIENCE:
    Data Scientist at Tech Corp (2020-2023)
    - Developed ML models
    - Analyzed large datasets
    """
    
    print("="*60)
    print("TESTING RESUME PREPROCESSOR")
    print("="*60)
    
    print("\n📄 Original Text:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    print("🧹 Cleaned Text:")
    cleaned = preprocessor.clean_text(sample_text)
    print(cleaned)
    print("\n" + "="*60 + "\n")
    
    print("✨ Fully Preprocessed Text:")
    processed = preprocessor.preprocess(sample_text)
    print(processed)
    print("\n" + "="*60)
    
    print("\n✅ Preprocessor test completed!")