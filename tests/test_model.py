from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=10.0, max_iter=10000, random_state=42)),
]).fit(X_tr, y_tr)

def test_accuracy_gate():
    acc = accuracy_score(y_te, pipe.predict(X_te))
    assert acc >= 0.95

def test_roc_auc_gate():
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert auc >= 0.98