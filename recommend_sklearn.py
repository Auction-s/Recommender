# src/recommend_sklearn.py
import os
import numpy as np
import pandas as pd
import pickle

ROOT = os.path.expanduser("~")
OUT_DIR = os.path.join(ROOT, "Documents", "Projects", "Bridge", "AI", "Recommender", "data", "parsed")

def load_model():
    user_factors = np.load(os.path.join(OUT_DIR, "user_factors.npy"))
    item_factors = np.load(os.path.join(OUT_DIR, "item_factors.npy"))
    with open(os.path.join(OUT_DIR, "mappings.pkl"), "rb") as f:
        maps = pickle.load(f)
    items = pd.read_csv(os.path.join(OUT_DIR, "items_small.csv"))
    return user_factors, item_factors, maps, items

def recommend(user_id, top_n=10):
    user_factors, item_factors, maps, items = load_model()
    user2idx = maps['user2idx']
    item2idx = maps['item2idx']

    if user_id not in user2idx:
        raise ValueError(f"User id {user_id} not in training data.")
    u_idx = user2idx[user_id]
    u_vec = user_factors[u_idx]  # shape (k,)
    scores = item_factors.dot(u_vec)  # shape (num_items,)
    top_idx = np.argsort(scores)[::-1][:top_n]

    # map top_idx back to item ids
    idx2item = {v:k for k,v in item2idx.items()}
    results = []
    for idx in top_idx:
        item_id = idx2item[idx]
        title_row = items[items['movie_id'] == item_id] if 'movie_id' in items.columns else items[items['item_id'] == item_id]
        title = title_row.iloc[0,1] if not title_row.empty else str(item_id)
        results.append((item_id, title, float(scores[idx])))
    return results

if __name__ == "__main__":
    uid = int(input("Enter user id (e.g. 1): ").strip() or "1")
    recs = recommend(uid, top_n=10)
    for iid, title, score in recs:
        print(f"{title}  (score={score:.3f})")
