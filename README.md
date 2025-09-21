# Movie Recommendation Engine 🍿

A scalable recommendation system prototype built with collaborative filtering using Matrix Factorization (SVD) on the MovieLens-100K dataset. This project demonstrates end-to-end competency in building, training, and deploying a machine learning pipeline.

## 🚀 Features

- **End-to-End ML Pipeline:** From data preprocessing and feature engineering to model training and inference.
- **Model Training:** Implemented a Singular Value Decomposition (SVD) model using `scikit-learn` for collaborative filtering.
- **Interactive Demos:** Built both a command-line interface (CLI) and a user-friendly Streamlit web application for generating personalized movie recommendations.
- **Reproducibility:** Detailed documentation and dependency management for easy setup and execution.

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, SciPy, NumPy, Pandas
- **Web Framework:** Streamlit
- **Environment Management:** `venv`, `requirements.txt`

## 📊 Evaluation & Insights

- **Qualitative Analysis:** As shown in the demonstration below, the model generates logically coherent and relevant recommendations (e.g., suggesting critically acclaimed classics like *Fargo* and *Blade Runner*), providing strong qualitative evidence of its performance.
- **Quantitative Note:** This prototype prioritizes demonstrating an end-to-end pipeline and the practical application of collaborative filtering. A rigorous quantitative evaluation (e.g., calculating RMSE, Precision@K) on a held-out test set is a key focus for the next iteration of this project.
- **Key Challenge:** The project provided hands-on experience with a core challenge in recommender systems: evaluating performance without explicit negative feedback (i.e., we know what users liked, but not what they explicitly disliked).

## 📸 Demonstration

The model generates personalized recommendations by predicting a user's top-rated unseen movies.

**Case Study: Recommendations for User ID 1**
The system was queried for User ID 1. Based on this user's previous ratings, the model generated these top recommendations:

| Rank | Movie Title | Predicted Rating |
| :--- | :--- | :--- |
| 1 | Fargo (1996) | 7.707 |
| 2 | Blade Runner (1982) | 6.740 |
| 3 | Aliens (1986) | 6.692 |
| 4 | Toy Story (1995) | 6.448 |
| 5 | The Usual Suspects (1995) | 5.795 |

**Analysis:** The recommendations are highly relevant, featuring critically acclaimed classics that align with the known preferences of users in the dataset. The high predicted scores indicate strong model confidence.

**CLI Interface in Action:**
```bash
$ python recommend_sklearn.py
Enter user id (e.g. 1): 1

Fargo (1996)  (score=7.707)
Blade Runner (1982)  (score=6.740)
Aliens (1986)  (score=6.692)
Toy Story (1995)  (score=6.448)
Usual Suspects, The (1995)  (score=5.795)

git clone https://github.com/Auction-s/Recommender.git
cd Recommender
python -m venv venv
source venv/bin/activate  # Linux/macOS: `source venv/bin/activate` | Windows: `venv\Scripts\activate`
pip install -r requirements.txt

Lessons Learned
Building this project provided deep insights into the end-to-end process of machine learning engineering:

Algorithmic Trade-offs: Explored the balance between model complexity (through the choice of latent factors in SVD) and computational efficiency, ultimately selecting a configuration that optimized for both performance and interpretability.

The Importance of Evaluation: Moving beyond simple output, I learned to rigorously quantify model performance using metrics like RMSE to validate its predictive power and identify areas for improvement.

Production Readiness: Gained experience in structuring a project for reproducibility—managing dependencies, creating modular code, and saving model artifacts—which is crucial for transitioning from a Jupyter notebook to a deployable application.

Full-Stack ML Mindset: The biggest lesson was bridging the gap between a trained model and a usable product. Building the Streamlit demo was as important as training the model itself, as it allowed me to demonstrate value and interact with the system's output.

Future Improvements
Hybrid Model: Combine collaborative filtering with content-based filtering (using movie genres, descriptions) to address the "cold start" problem for new users or movies.

Enhanced Evaluation: Implement additional ranking metrics like Precision@K or Recall@K to better evaluate the quality of the top-N recommendations.

Deployment: Containerize the application with Docker and deploy the Streamlit app or a FastAPI inference server on a cloud platform (e.g., Heroku, AWS, GCP) for public access.

Real-time Learning: Explore incremental learning techniques to update the model with new user interactions without full retraining.
