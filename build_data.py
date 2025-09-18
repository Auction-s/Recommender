import os
import pandas as pd

# Define paths
#C:\Users\LENOVO\Documents\Projects\Bridge\AI\Recommender\data\ml-100k\ml-100k
ROOT = os.path.expanduser('~')
ml_dir = os.path.join(ROOT, 'Documents', 'Projects', 'Bridge', 'AI', 'Recommender', 'data', 'ml-100k', 'ml-100k')
rating_dir = os.path.join(ml_dir, 'u.data')
item_dir = os.path.join(ml_dir, 'u.item')
out_dir = os.path.join(ROOT, 'Documents', 'Projects', 'Bridge', 'AI', 'Recommender', 'data', 'parsed')

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

def build_ratings():
    """
    Load the ratings file (u.data) and save as CSV.
    Columns: user_id, item_id, rating, timestamp
    """
    ratings = pd.read_csv(
        rating_dir,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    ratings.to_csv(os.path.join(out_dir, 'ratings.csv'), index=False)
    print(f"✅ Ratings saved to {os.path.join(out_dir, 'ratings.csv')}")

def build_items():
    """
    Load the item file (u.item) and save as CSV.
    Columns: movie_id, movie_title, release_date, video_release_date, imdb_url, genres...
    """
    # MovieLens 100k `u.item` has 24 columns
    item_cols = [
        'movie_id', 'movie_title', 'release_date', 'video_release_date',
        'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
        'War', 'Western'
    ]
    items = pd.read_csv(
        item_dir,
        sep='|',
        names=item_cols,
        encoding='latin-1'
    )
    items.to_csv(os.path.join(out_dir, 'items.csv'), index=False)
    print(f"✅ Items saved to {os.path.join(out_dir, 'items.csv')}")

if __name__ == "__main__":
    build_ratings()
    build_items()
    print("🎉 Data preprocessing complete!")
