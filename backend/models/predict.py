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


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN GROUPS
# Roles in the same group are considered "same domain".
# Rank 2 and 3 will ONLY be picked from the same domain as rank 1.
# ──────────────────────────────────────────────────────────────────────────────
DOMAIN_GROUPS = {
    # Tech & Engineering
    'INFORMATION-TECHNOLOGY'   : 'tech',
    'ENGINEERING'              : 'tech',
    'DATA-ANALYST'             : 'tech',
    'DATA-SCIENCE'             : 'tech',
    'JAVA-DEVELOPER'           : 'tech',
    'PYTHON-DEVELOPER'         : 'tech',
    'DEVOPS'                   : 'tech',
    'DOTNET-DEVELOPER'         : 'tech',
    'DATABASE'                 : 'tech',
    'ETL-DEVELOPER'            : 'tech',
    'NETWORK-SECURITY-ENGINEER': 'tech',
    'SAP-DEVELOPER'            : 'tech',
    'REACT-DEVELOPER'          : 'tech',
    'TESTING'                  : 'tech',

    # Engineering & Trades
    'CIVIL-ENGINEER'           : 'engineering',
    'MECHANICAL-ENGINEER'      : 'engineering',
    'ELECTRICAL-ENGINEERING'   : 'engineering',
    'CONSTRUCTION'             : 'engineering',
    'AUTOMOBILE'               : 'engineering',
    'AVIATION'                 : 'engineering',

    # Creative & Arts
    'ARTS'                     : 'creative',
    'APPAREL'                  : 'creative',
    'DESIGNER'                 : 'creative',
    'DIGITAL-MEDIA'            : 'creative',
    'WEB-DESIGNING'            : 'creative',

    # Business & Management
    'BUSINESS-DEVELOPMENT'   : 'business',
    'CONSULTANT'             : 'business',
    'SALES'                  : 'business',
    'PUBLIC-RELATIONS'       : 'business',

    # Finance & Legal
    'ACCOUNTANT'             : 'finance',
    'FINANCE'                : 'finance',
    'BANKING'                : 'finance',
    'ADVOCATE'               : 'finance',

    # People & Admin
    'HR'                     : 'people',
    'BPO'                    : 'people',
    'TEACHER'                : 'people',

    # Health & Lifestyle
    'HEALTHCARE'             : 'health',
    'FITNESS'                : 'health',
    'FOOD-AND-BEVERAGES'     : 'health',
    'CHEF'                   : 'health',
    'AGRICULTURE'            : 'health',




}


class ResumePredictor:
    """
    Loads trained model and predicts job role from resume text.
    Top 3 roles are always from the SAME DOMAIN as the top prediction,
    so you never get unrelated roles like APPAREL for a UX Designer.
    """

    # Minimum probability for rank 2/3 to be shown
    MIN_CONFIDENCE = 0.05

    def __init__(self):
        self.preprocessor          = ResumePreprocessor()
        self.model                 = None
        self.vectorizer            = None
        self.label_encoder         = None
        self.inverse_label_encoder = None
        self._feature_names        = None
        self._keyword_cache        = {}
        self.load_model()

    # ----------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------
    def load_model(self):
        """Load saved model, vectorizer, and label encoder."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir  = os.path.join(os.path.dirname(current_dir), 'saved_models')

        self.model         = joblib.load(os.path.join(models_dir, 'model.pkl'))
        self.vectorizer    = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))

        # Support both dict and list formats for label_encoder
        if isinstance(self.label_encoder, list):
            self.label_encoder = {role: idx for idx, role in enumerate(self.label_encoder)}

        self.inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}

        try:
            self._feature_names = self.vectorizer.get_feature_names_out()
        except AttributeError:
            self._feature_names = np.array(self.vectorizer.get_feature_names())

        print("Model loaded successfully!")

    # ----------------------------------------------------------------
    # PREDICT  (domain-aware top 3)
    # ----------------------------------------------------------------
    def predict(self, resume_text: str) -> dict:
        """
        Predict job role from resume text.

        Returns:
            dict with:
                predicted_role  — top predicted role
                confidence      — confidence of top prediction
                top_3_roles     — list of (role, probability) tuples,
                                  ALL from the same domain as predicted_role
        """
        cleaned_text  = self.preprocessor.preprocess(resume_text)
        features      = self.vectorizer.transform([cleaned_text])
        probabilities = self.model.predict_proba(features)[0]

        # Sort ALL roles by probability descending
        sorted_indices = probabilities.argsort()[::-1]

        # Top prediction
        top_idx        = sorted_indices[0]
        predicted_role = self.inverse_label_encoder[top_idx]
        confidence     = float(probabilities[top_idx])

        # Find domain of top prediction
        top_domain = DOMAIN_GROUPS.get(predicted_role, None)

        # Strictly collect top 3 from same domain only — no fallback to other domains
        same_domain_roles = []
        for idx in sorted_indices:
            role = self.inverse_label_encoder[idx]
            prob = float(probabilities[idx])
            if DOMAIN_GROUPS.get(role) == top_domain:
                same_domain_roles.append((role, prob))

        top_3_roles = same_domain_roles[:3]

        # If domain has fewer than 3 roles total, pad with N/A
        while len(top_3_roles) < 3:
            top_3_roles.append(('N/A', 0.0))

        # Low confidence = model is unsure (likely an unseen role)
        LOW_CONFIDENCE_THRESHOLD = 0.30
        is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

        return {
            'predicted_role'   : predicted_role,
            'confidence'       : confidence,
            'top_3_roles'      : top_3_roles,
            'domain'           : top_domain,
            'low_confidence'   : is_low_confidence,
            'low_conf_message' : (
                "We couldn't confidently match your resume to a known role. "
                "This may be because your role (e.g. Data Analyst, Product Manager) "
                "isn't in our current model. The closest matches are shown below."
            ) if is_low_confidence else None,
        }

    # ----------------------------------------------------------------
    # ML-LEARNED KEYWORDS
    # ----------------------------------------------------------------
    def get_top_keywords_for_role(self, role: str, n: int = 40) -> list:
        """
        Returns the top-n keywords the ML model learned for a given role.
        Works with LogisticRegression, LinearSVC, NaiveBayes, RandomForest.
        """
        cache_key = f"{role}_{n}"
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]

        class_idx = self._resolve_class_idx(role)
        if class_idx is None:
            print(f"[get_top_keywords_for_role] Role '{role}' not found.")
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
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(model, Pipeline):
                return model.steps[-1][1]
        except ImportError:
            pass
        return model

    def _extract_top_features(self, model, class_idx: int, n: int) -> list:
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

    # Test 1: UX Designer
    ux_resume = """
    UX Designer with 5 years of experience.
    Skills: Figma, Sketch, Adobe XD, User Research, Wireframing,
    Prototyping, Usability Testing, Interaction Design, Design Thinking.
    Experience:
    Senior UX Designer at Product Co (2021-2024)
    - Conducted user research and interviews
    - Created wireframes and prototypes in Figma
    - Ran usability testing sessions
    """

    # Test 2: Software Developer
    dev_resume = """
    Software Developer with 5 years in web development.
    Skills: React.js, Node.js, Python, MongoDB, PostgreSQL, Docker, AWS, Git
    Experience:
    Senior Developer at Tech Corp (2020-2024)
    - Built full-stack web applications
    - Designed RESTful APIs and microservices
    """

    for label, resume in [("UX Designer", ux_resume), ("Software Developer", dev_resume)]:
        print(f"\n{'='*55}")
        print(f"TEST: {label}")
        print('='*55)
        result = predictor.predict(resume)
        print(f"Domain   : {result['domain']}")
        print(f"Top Role : {result['predicted_role']} ({result['confidence']*100:.1f}%)")
        print("Top 3    :")
        for i, (role, prob) in enumerate(result['top_3_roles'], 1):
            print(f"  {i}. {role}: {prob*100:.1f}%")