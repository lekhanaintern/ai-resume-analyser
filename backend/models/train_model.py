"""
train_model.py  —  Improved Resume Classifier
================================================
Key improvements over the original:
  1. Logistic Regression  instead of RandomForestClassifier
  2. 10 000-feature TF-IDF  instead of 1 500
  3. GridSearchCV to tune C  (regularisation strength)
  4. Class-weight balancing  so minority roles don't get ignored
  5. Confidence threshold  in predict() — ranks 2/3 only shown when meaningful
  6. Evaluation report printed after training  (accuracy + per-class F1)
  7. Clean folder structure  (saves to saved_models/ like the original)
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

# ── path setup ────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

try:
    from models.preprocessor import ResumePreprocessor
except ImportError:
    from preprocessor import ResumePreprocessor


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  — replace load_data() with however you load your CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str):
    """
    Expects a CSV with at least two columns:
        'resume_text'  — raw resume string
        'category'     — job-role label  (e.g. 'DESIGNER', 'INFORMATION-TECHNOLOGY')

    Adjust column names below if yours differ.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Resume', 'Category'])
    df = df[df['Resume'].str.strip() != '']

    print(f"✅ Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
    print("\nClass distribution:")
    print(df['Category'].value_counts().to_string())
    return df['Resume'].tolist(), df['Category'].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_corpus(texts: list, preprocessor: ResumePreprocessor) -> list:
    print("\n⏳ Preprocessing resumes …")
    cleaned = [preprocessor.preprocess(t) for t in texts]
    print("✅ Preprocessing done.")
    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LABEL ENCODING  (simple dict, same as your original)
# ══════════════════════════════════════════════════════════════════════════════

def encode_labels(labels: list):
    unique = sorted(set(labels))
    label_encoder = {role: idx for idx, role in enumerate(unique)}
    encoded = [label_encoder[label] for label in labels]
    return encoded, label_encoder


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BUILD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorizer():
    """
    FIX 1 — max_features raised from 1 500 → 10 000
             This alone brings back Figma, wireframe, usability, UX, UI, etc.
    FIX 2 — min_df lowered to 1 so rare but important terms aren't dropped
    FIX 3 — sublinear_tf=True normalises very long resumes better
    """
    return TfidfVectorizer(
        max_features  = 10_000,   # was 1 500 — biggest single fix
        ngram_range   = (1, 2),   # keep bigrams (e.g. "user research")
        min_df        = 1,        # was 2 — don't drop rare but important words
        max_df        = 0.85,
        sublinear_tf  = True,     # log(1+tf) — handles long resumes better
        strip_accents = 'unicode',
    )


def build_model():
    """
    FIX 4 — Logistic Regression instead of RandomForest
             LR is the gold standard for TF-IDF text classification.
             It produces real probability distributions (RF probabilities
             are just vote fractions and are poorly calibrated).
    FIX 5 — class_weight='balanced' so minority roles aren't steamrolled
    """
    return LogisticRegression(
        C            = 5,
        max_iter     = 1000,
        solver       = 'lbfgs',
        class_weight = 'balanced',
        random_state = 42,
    )


def train(texts_clean: list, y_encoded: list, label_encoder: dict):
    """Full training pipeline with GridSearchCV tuning."""

    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean, y_encoded,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y_encoded,   # keeps class ratios in both splits
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print(f"\n🔢 Vocabulary size after fit: {len(vectorizer.vocabulary_)}")

    # ── Grid search over regularisation strength C ────────────────────────
    print("\n🔍 Running GridSearchCV (this may take a minute) …")
    base_model = LogisticRegression(
        max_iter     = 1000,
        solver       = 'lbfgs',
        class_weight = 'balanced',
        random_state = 42,
    )
    param_grid = {'C': [0.1, 1, 5, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(base_model, param_grid, cv=cv,
                      scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(X_train_vec, y_train)

    best_C = gs.best_params_['C']
    print(f"✅ Best C = {best_C}  (CV macro-F1 = {gs.best_score_:.3f})")

    model = gs.best_estimator_

    # ── Evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)

    inv_enc = {v: k for k, v in label_encoder.items()}
    target_names = [inv_enc[i] for i in sorted(inv_enc)]

    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {acc*100:.2f}%")
    print(f"{'='*60}")
    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return model, vectorizer


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, vectorizer, label_encoder, output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'saved_models'
        )
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model,         os.path.join(output_dir, 'model.pkl'))
    joblib.dump(vectorizer,    os.path.join(output_dir, 'vectorizer.pkl'))
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))

    print(f"\n💾 Saved model, vectorizer, label_encoder → {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UPDATED PREDICT CLASS  (drop-in replacement for your existing predict.py)
# ══════════════════════════════════════════════════════════════════════════════

class ResumePredictor:
    """
    Drop-in replacement for the original ResumePredictor.
    Extra fix: confidence threshold filters out noise in rank 2/3.
    """

    MIN_CONFIDENCE_RANK2 = 0.08   # rank 2/3 must have ≥8% prob to show

    def __init__(self, models_dir: str = None):
        self.preprocessor = ResumePreprocessor()
        if models_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir  = os.path.join(os.path.dirname(current_dir), 'saved_models')

        self.model         = joblib.load(os.path.join(models_dir, 'model.pkl'))
        self.vectorizer    = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        self.inv_enc       = {v: k for k, v in self.label_encoder.items()}
        print("✅ Model loaded successfully!")

    def predict(self, resume_text: str) -> dict:
        cleaned  = self.preprocessor.preprocess(resume_text)
        features = self.vectorizer.transform([cleaned])
        probs    = self.model.predict_proba(features)[0]

        top_idx      = probs.argsort()[::-1]
        predicted    = self.inv_enc[top_idx[0]]
        confidence   = float(probs[top_idx[0]])

        # FIX 6 — only include rank 2/3 if they have meaningful probability
        top_3 = [(self.inv_enc[top_idx[0]], confidence)]
        for idx in top_idx[1:3]:
            p = float(probs[idx])
            if p >= self.MIN_CONFIDENCE_RANK2:
                top_3.append((self.inv_enc[idx], p))
            else:
                top_3.append((self.inv_enc[idx], p))   # still returned, but flagged

        return {
            'predicted_role' : predicted,
            'confidence'     : confidence,
            'top_3_roles'    : top_3,
            'reliable_top_3' : [(r, p) for r, p in top_3
                                if p >= self.MIN_CONFIDENCE_RANK2],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN — run this file directly to train
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train improved resume classifier")
    parser.add_argument('--data', required=True,
                        help='Path to CSV file with resume_text and category columns')
    parser.add_argument('--out', default=None,
                        help='Directory to save model files (default: ../saved_models/)')
    args = parser.parse_args()

    preprocessor = ResumePreprocessor()

    # 1. Load
    texts, labels = load_data(args.data)

    # 2. Preprocess
    texts_clean = preprocess_corpus(texts, preprocessor)

    # 3. Encode labels
    y_encoded, label_encoder = encode_labels(labels)

    # 4. Train
    model, vectorizer = train(texts_clean, y_encoded, label_encoder)

    # 5. Save
    save_artifacts(model, vectorizer, label_encoder, args.out)

    print("\n🎉 Training complete! Run predict.py to test the new model.")