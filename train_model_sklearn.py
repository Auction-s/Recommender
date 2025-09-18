# src/train_model_sklearn.py
"""
Train a simple matrix-factorization-like recommender using SVD (scipy.sparse.linalg.svds).
Produces two matrices: user_factors.npy and item_factors.npy and saves item mapping.
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import pickle

ROOT = os.path.expanduser("~")
OUT_DIR = os.path.join(ROOT, "Documents", "Projects", "Bridge", "AI", "Recommender", "data", "parsed")
RATINGS_CSV = os.path.join(OUT_DIR, "ratings.csv")
ITEMS_CSV = os.path.join(OUT_DIR, "items.csv")

K = 50  # number of latent factors

def load_data():
    ratings = pd.read_csv(RATINGS_CSV)
    items = pd.read_csv(ITEMS_CSV)
    return ratings, items

def build_matrix(ratings):
    """
    Build user-item sparse matrix and mappings.
    """
    user_ids = ratings['user_id'].unique()
    item_ids = ratings['item_id'].unique()
    user2idx = {u:i for i,u in enumerate(sorted(user_ids))}
    item2idx = {i:j for j,i in enumerate(sorted(item_ids))}

    rows = ratings['user_id'].map(user2idx).to_numpy()
    cols = ratings['item_id'].map(item2idx).to_numpy()
    vals = ratings['rating'].to_numpy().astype(float)

    mat = coo_matrix((vals, (rows, cols)), shape=(len(user2idx), len(item2idx)))
    return mat, user2idx, item2idx

def train_svd(mat, k=K):
    """
    Compute truncated SVD on the user-item matrix (centring not performed).
    Returns user_factors (num_users x k) and item_factors (num_items x k).
    """
    print("Computing truncated SVD, k=", k)
    # svds returns (u, s, vt)
    u, s, vt = svds(mat, k=k)
    # svds returns ascending singular values; reverse
    u = u[:, ::-1]
    s = s[::-1]
    vt = vt[::-1, :]
    # convert to factors
    user_factors = u.dot(np.diag(np.sqrt(s)))
    item_factors = vt.T.dot(np.diag(np.sqrt(s)))
    return user_factors, item_factors

def save_model(user_factors, item_factors, user2idx, item2idx, items_df, out_dir=OUT_DIR):
    print("Saving model artifacts...")
    os.makedirs(out_dir, exist_ok=True)
    print("Saving user factors shape:", user_factors.shape)
    print("Saving item factors shape:", item_factors.shape)

    np.save(os.path.join(out_dir, "user_factors.npy"), user_factors)
    np.save(os.path.join(out_dir, "item_factors.npy"), item_factors)
    with open(os.path.join(out_dir, "mappings.pkl"), "wb") as f:
        pickle.dump({"user2idx": user2idx, "item2idx": item2idx}, f)
    # Save small items_df for titles
    items_df.to_csv(os.path.join(out_dir, "items_small.csv"), index=False)
    print("✅ Saved model artifacts to", out_dir)


if __name__ == "__main__":
    ratings, items = load_data()
    mat, user2idx, item2idx = build_matrix(ratings)
    user_factors, item_factors = train_svd(mat, k=K)
    save_model(user_factors, item_factors, user2idx, item2idx, items)
    print("Done.")
