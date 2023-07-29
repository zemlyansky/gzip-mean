import gzip
import numpy as np
import pickle
from functools import lru_cache
from sklearn.metrics import accuracy_score
from tqdm import tqdm

N_SAMPLES = 10000
N_POS = N_NEG = 500

# Load dataset
with open(f"sentiment-dataset-{N_SAMPLES}.pickle", "rb") as f:
    dataset = pickle.load(f)
X_train, y_train, X_test, y_test = dataset
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

# Sample positive and negative examples
i_pos = np.array(y_train) == 1
X_train_pos = np.random.choice(np.array(X_train)[i_pos], N_POS, replace=False)
X_train_neg = np.random.choice(np.array(X_train)[~i_pos], N_NEG, replace=False)

@lru_cache(maxsize=N_SAMPLES)
def size(x: str) -> int:
    """Return the size of a string in bytes after compression with gzip"""
    return len(gzip.compress(x.encode('utf-8')))

def score(x1: str, x2: str) -> float:
    """Return the combined compression score"""
    return 1 - size(x1 + ' ' + x2) / (size(x1) + size(x2))

def predict(X: np.ndarray) -> np.ndarray:
    """Classify a list of strings as positive or negative"""
    y = []
    for x in tqdm(X):
        pos_scores = [score(x, x_pos) for x_pos in X_train_pos]
        neg_scores = [score(x, x_neg) for x_neg in X_train_neg]
        y.append(1 if np.mean(pos_scores) > np.mean(neg_scores) else -1)
    return np.array(y)

y_pred = predict(X_test)
print('accuracy:', accuracy_score(y_test, y_pred))