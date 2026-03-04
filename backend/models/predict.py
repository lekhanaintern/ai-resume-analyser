import joblib
import os
import sys
import numpy as np

# Fix: Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from models.preprocessor import ResumePreprocessor
except ImportError:
    from preprocessor import ResumePreprocessor


class ResumePredictor:
    """
    Loads trained model and predicts job role from resume text.
    Also exposes get_top_keywords_for_role() which extracts the keywords
    the ML model ACTUALLY LEARNED are most important for each role.
    """

    def __init__(self):
        self.preprocessor          = ResumePreprocessor()
        self.model                 = None
        self.vectorizer            = None
        self.label_encoder         = None
        self.inverse_label_encoder = None
        self._feature_names        = None   # TF-IDF vocabulary array
        self._keyword_cache        = {}     # cache: "role_n" -> [keywords]
        self.load_model()

    # ----------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------
    def load_model(self):
        """Load saved model, vectorizer, and label encoder."""
        current_dir   = os.path.dirname(os.path.abspath(__file__))
        models_dir    = os.path.join(os.path.dirname(current_dir), 'saved_models')

        self.model         = joblib.load(os.path.join(models_dir, 'model.pkl'))
        self.vectorizer    = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))

        # inverse mapping: encoded-int -> role-name
        self.inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}

        # Cache TF-IDF feature names once at load time
        try:
            self._feature_names = self.vectorizer.get_feature_names_out()
        except AttributeError:
            # older sklearn
            self._feature_names = np.array(self.vectorizer.get_feature_names())

        print("Model loaded successfully!")

    # ----------------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------------
    def predict(self, resume_text):
        """
        Predict job role from resume text.

        Returns:
            dict with predicted_role, confidence, top_3_roles
        """
        cleaned_text  = self.preprocessor.preprocess(resume_text)
        features      = self.vectorizer.transform([cleaned_text])
        prediction    = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        predicted_role = self.inverse_label_encoder[prediction]
        confidence     = probabilities[prediction]

        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_roles   = [
            (self.inverse_label_encoder[idx], float(probabilities[idx]))
            for idx in top_3_indices
        ]

        return {
            'predicted_role': predicted_role,
            'confidence'    : float(confidence),
            'top_3_roles'   : top_3_roles,
        }

    # ----------------------------------------------------------------
    # ML-LEARNED KEYWORDS
    # ----------------------------------------------------------------
    def get_top_keywords_for_role(self, role: str, n: int = 40) -> list:
        """
        Ask the trained ML model which words it learned are most important
        for a given job role — directly from classifier weights/log-probs.

        Works with:
          LogisticRegression  -> coef_[class_idx]
          LinearSVC           -> coef_[class_idx]
          SGDClassifier       -> coef_[class_idx]
          MultinomialNB       -> feature_log_prob_[class_idx]
          ComplementNB        -> feature_log_prob_[class_idx]
          BernoulliNB         -> feature_log_prob_[class_idx]
          RandomForestClassifier -> feature_importances_ (global proxy)
          Pipeline wrappers   -> unwrapped automatically

        Args:
            role: Role name as it appears in label_encoder
            n   : Number of keywords to return (default 40)

        Returns:
            list of cleaned, display-ready keyword strings
        """
        cache_key = f"{role}_{n}"
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]

        class_idx = self._resolve_class_idx(role)
        if class_idx is None:
            print(f"[get_top_keywords_for_role] Role '{role}' not found in label_encoder.")
            return []

        if self._feature_names is None or len(self._feature_names) == 0:
            print("[get_top_keywords_for_role] Feature names unavailable.")
            return []

        model    = self._unwrap_pipeline(self.model)
        raw      = self._extract_top_features(model, class_idx, n * 3)
        keywords = self._clean_keywords(raw, n)

        self._keyword_cache[cache_key] = keywords
        print(f"[ML keywords] '{role}' => {len(keywords)} terms: {keywords[:6]}")
        return keywords

    # ----------------------------------------------------------------
    # PRIVATE HELPERS
    # ----------------------------------------------------------------
    def _resolve_class_idx(self, role: str):
        """Find encoded class index for a role name — tries exact, case-insensitive, partial."""
        if role in self.label_encoder:
            return self.label_encoder[role]
        role_l = role.lower().strip()
        for k, v in self.label_encoder.items():
            if k.lower().strip() == role_l:
                return v
        for k, v in self.label_encoder.items():
            if role_l in k.lower() or k.lower() in role_l:
                return v
        return None

    def _unwrap_pipeline(self, model):
        """Unwrap sklearn Pipeline to get the actual classifier."""
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(model, Pipeline):
                return model.steps[-1][1]
        except ImportError:
            pass
        return model

    def _extract_top_features(self, model, class_idx: int, n: int) -> list:
        """Extract top-n feature names using the correct attribute for each model type."""
        name = type(model).__name__
        try:
            if name in ('LogisticRegression', 'LogisticRegressionCV'):
                w = np.asarray(model.coef_[class_idx]).ravel()
                return [self._feature_names[i] for i in w.argsort()[-n:][::-1]]

            elif name in ('LinearSVC', 'SGDClassifier'):
                coef = np.asarray(model.coef_)
                w    = coef[class_idx] if coef.ndim == 2 else (
                    coef[0] if class_idx == 1 else -coef[0]
                )
                return [self._feature_names[i] for i in w.argsort()[-n:][::-1]]

            elif name in ('MultinomialNB', 'ComplementNB', 'BernoulliNB', 'CategoricalNB'):
                w = np.asarray(model.feature_log_prob_[class_idx])
                return [self._feature_names[i] for i in w.argsort()[-n:][::-1]]

            elif name in ('RandomForestClassifier', 'ExtraTreesClassifier',
                          'GradientBoostingClassifier'):
                w = np.asarray(model.feature_importances_)
                return [self._feature_names[i] for i in w.argsort()[-n:][::-1]]

            else:
                print(f"[_extract_top_features] Unknown model '{name}' — trying generic fallback.")
                if hasattr(model, 'coef_'):
                    coef = np.asarray(model.coef_)
                    row  = coef[class_idx] if coef.ndim == 2 else coef[0]
                    return [self._feature_names[i] for i in row.argsort()[-n:][::-1]]
                if hasattr(model, 'feature_importances_'):
                    w = np.asarray(model.feature_importances_)
                    return [self._feature_names[i] for i in w.argsort()[-n:][::-1]]

        except Exception as e:
            print(f"[_extract_top_features] Error: {e}")
        return []

    def _clean_keywords(self, raw: list, n: int) -> list:
        """Remove noise tokens, stopwords; title-case; deduplicate."""
        STOPWORDS = {
            'the','and','for','with','that','this','from','have','been','will',
            'are','was','were','had','has','can','not','but','all','also','its',
            'our','their','they','which','when','your','you','any','may','more',
            'other','some','such','than','then','there','these','those','into',
            'over','under','about','after','before','each','very','just','through',
            'during','between','both','few','most','per','too','because','while',
            'use','used','using','work','working','worked','help','make','good',
        }
        seen, clean = set(), []
        for kw in raw:
            kw = kw.strip()
            if len(kw) < 2:
                continue
            if kw.isdigit():
                continue
            if kw.lower() in STOPWORDS:
                continue
            if sum(c.isalpha() for c in kw) / max(len(kw), 1) < 0.5:
                continue
            display = kw.title() if ' ' in kw else (
                kw if any(c.isupper() for c in kw[1:]) else kw.capitalize()
            )
            key = display.lower()
            if key not in seen:
                seen.add(key)
                clean.append(display)
            if len(clean) >= n:
                break
        return clean


# ----------------------------------------------------------------
# QUICK TEST
# ----------------------------------------------------------------
if __name__ == "__main__":
    predictor = ResumePredictor()

    sample = """
    JOHN DOE  |  john@example.com  |  +1-555-1234
    PROFESSIONAL SUMMARY
    Software Developer with 5 years in web development.
    SKILLS
    React.js, HTML5, CSS3, JavaScript, Node.js, Python, MongoDB, Git
    EXPERIENCE
    Senior Web Developer - Tech Corp (2020-2024)
    - Developed full-stack apps using React and Node.js
    - Built RESTful APIs, managed PostgreSQL databases
    EDUCATION
    B.Sc Computer Science - State University 2019
    """

    result = predictor.predict(sample)
    print(f"Predicted: {result['predicted_role']} ({result['confidence']*100:.1f}%)")

    for role, prob in result['top_3_roles']:
        kws = predictor.get_top_keywords_for_role(role, n=15)
        print(f"\n{role} ({prob*100:.1f}%):")
        print("  " + ", ".join(kws))