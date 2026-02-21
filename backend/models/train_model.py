import joblib
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from models.preprocessor import ResumePreprocessor
except ImportError:
    from preprocessor import ResumePreprocessor

class ResumePredictor:
    """
    Loads trained model and predicts job role from resume text
    """
    
    def __init__(self):
        self.preprocessor = ResumePreprocessor()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.inverse_label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load saved model, vectorizer, and label encoder"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), 'saved_models')
        
        model_path = os.path.join(models_dir, 'model.pkl')
        vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        
        # Load all components
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Create inverse mapping
        self.inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        
        print("✅ Model loaded successfully!")
    
    def predict(self, resume_text):
        """
        Predict job role from resume text
        
        Args:
            resume_text (str): Raw resume text
            
        Returns:
            dict: {
                'predicted_role': str,
                'confidence': float,
                'top_3_roles': list of tuples (role, probability)
            }
        """
        # Step 1: Preprocess the text
        cleaned_text = self.preprocessor.preprocess(resume_text)
        
        # Step 2: Convert to TF-IDF features
        features = self.vectorizer.transform([cleaned_text])
        
        # Step 3: Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Step 4: Get predicted role
        predicted_role = self.inverse_label_encoder[prediction]
        confidence = probabilities[prediction]
        
        # Step 5: Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_roles = [
            (self.inverse_label_encoder[idx], float(probabilities[idx]))
            for idx in top_3_indices
        ]
        
        return {
            'predicted_role': predicted_role,
            'confidence': float(confidence),
            'top_3_roles': top_3_roles
        }

# Test the predictor
if __name__ == "__main__":
    predictor = ResumePredictor()
    
    # Test with sample resume
    sample_resume = """
    JOHN DOE
    Email: john@example.com
    Phone: +1-555-1234
    
    PROFESSIONAL SUMMARY
    Experienced Software Developer with 5+ years in web development.
    Proficient in React, Node.js, Python, and MongoDB.
    
    SKILLS
    - Frontend: React.js, HTML5, CSS3, JavaScript, TypeScript
    - Backend: Node.js, Express, Python, Flask
    - Database: MongoDB, PostgreSQL, MySQL
    - Tools: Git, Docker, AWS
    
    EXPERIENCE
    Senior Web Developer - Tech Corp (2020-2024)
    - Developed full-stack web applications using React and Node.js
    - Built RESTful APIs and microservices
    - Worked with MongoDB and PostgreSQL databases
    
    EDUCATION
    Bachelor of Computer Science
    """
    
    print("\n" + "="*60)
    print("TESTING RESUME PREDICTOR")
    print("="*60)
    
    result = predictor.predict(sample_resume)
    
    print(f"\n🎯 Predicted Role: {result['predicted_role']}")
    print(f"📊 Confidence: {result['confidence']*100:.2f}%")
    print("\n🏆 Top 3 Predictions:")
    for i, (role, prob) in enumerate(result['top_3_roles'], 1):
        print(f"  {i}. {role}: {prob*100:.2f}%")