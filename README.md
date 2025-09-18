# Recommender 🍿📚

**One-line:** A compact recommendation engine built on MovieLens-100k using collaborative filtering (SVD) — demonstrates skills relevant to Amazon-style recommender systems.

---

## 📌 Overview
This project builds a small, reproducible recommendation pipeline:
- Preprocesses the MovieLens-100k dataset.
- Trains a lightweight SVD-based model (matrix factorization).
- Saves latent factor artifacts.
- Provides both a **CLI demo** and an optional **Streamlit demo** for recommendations.

It’s designed as a concise portfolio piece demonstrating practical recommender engineering — directly tied to Amazon’s core systems.

---

## ✅ What I built
- Data parsing & preprocessing (`src/build_data.py`).
- Training with truncated SVD (`src/train_model_sklearn.py`).
- CLI recommendation demo (`src/recommend_sklearn.py`).
- Interactive Streamlit demo (`src/app_streamlit.py`).
- A compact items file (`data/parsed/items_small.csv`) so recommendations are human-readable.

---

## 🛠 Tech stack
- **Python 3.8+**
- **numpy, pandas, scipy, scikit-learn**
- **Streamlit** (for interactive demo)

---

## ⚡ Quickstart
1. Clone the repo:
```bash
git clone https://github.com/Auction-s/Recommender.git
cd Recommender

python -m venv recommender_env
recommender_env\Scripts\activate   # Windows
# or: source recommender_env/bin/activate   # Mac/Linux

pip install -r requirements.txt

python src/train_model_sklearn.py

---

## ✨ Personal Growth — What I learned building this project

Working on this mini recommender taught me both technical skills and product thinking:

**Technical skills**
- Hands-on experience with data ingestion and preprocessing (parsing MovieLens files, cleaning, saving CSVs).
- Building a scalable recommender pipeline using matrix factorization (SVD) with `scipy` and `numpy`.
- Working with sparse matrices (`scipy.sparse`) for efficient memory use.
- Saving and loading model artifacts (`.npy`, `.pkl`) for reproducible demos.
- Creating an interactive demo with **Streamlit** for quick stakeholder-friendly demos.

**Engineering & project skills**
- Organizing reproducible projects: separate `src/`, `data/`, `requirements.txt`, and `README.md`.
- Using virtual environments to isolate dependencies.
- Learning practical trade-offs (train vs deliverable artifact sizes, when to include artifacts in repo).
- Writing clear documentation so others can reproduce and evaluate the work.

**Challenges & lessons**
- Installing some ML packages on Windows can require build tools; learned fallback approaches (use binary wheels, or switch to pure-Python alternatives).
- Emphasized reproducibility — always include a simple `train` script and a `requirements.txt` so reviewers can run the project.
- Learned how to present model results qualitatively (demo + screenshots) when quantitative evaluation is limited.

**Next steps I plan to take**
- Add evaluation metrics (RMSE, Precision@k) and a short analysis notebook.
- Build a deployable API (FastAPI) and optionally deploy a demo on a cloud provider (Heroku/Render/AWS).
- Extend to hybrid approaches (content + collaborative filtering) and fine-tune on domain-specific data.

---


update README with project description
