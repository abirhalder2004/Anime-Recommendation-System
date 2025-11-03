from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from functools import wraps
from time import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
app.config['JIKAN_API_URL'] = os.getenv('JIKAN_API_URL', 'https://api.jikan.moe/v4')
app.config['RATE_LIMIT'] = float(os.getenv('RATE_LIMIT', '3'))  # requests per second
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Rate limiting decorator for Jikan API
def jikan_rate_limit(func):
    last_call = 0
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal last_call
        elapsed = time() - last_call
        wait_time = 1.0 / app.config['RATE_LIMIT']
        
        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)
        
        last_call = time()
        return func(*args, **kwargs)
    return wrapper

# Load and preprocess data (only once at startup)
def load_data():
    logger.info("Loading and preprocessing data...")
    anime_df = pd.read_csv("data/anime.csv")
    rating_df = pd.read_csv("data/rating.csv")

    # Data cleaning
    anime_df.dropna(subset=["genre", "type"], inplace=True)
    anime_df["genre"] = anime_df["genre"].fillna("Unknown")
    anime_df["rating"] = anime_df["rating"].fillna(anime_df["rating"].median())
    rating_df = rating_df[rating_df["rating"] != -1]

    # Feature engineering
    anime_df["combined_features"] = anime_df["genre"] + " " + anime_df["type"] + " " + anime_df["name"]

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(anime_df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(anime_df.index, index=anime_df["name"]).drop_duplicates()

    return anime_df, rating_df, cosine_sim, indices

# Load data at startup
anime_df, rating_df, cosine_sim, indices = load_data()

@jikan_rate_limit
def get_anime_image_url(title):
    try:
        url = f"{app.config['JIKAN_API_URL']}/anime?q={title}&limit=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["data"]:
            return data["data"][0]["images"]["jpg"]["image_url"]
    except Exception as e:
        logger.error(f"Image fetch failed for {title}: {str(e)}")
    return "https://via.placeholder.com/150?text=No+Image"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        anime_name = request.form.get("anime_name", "").strip()
        types_selected = request.form.getlist("types")
        page = int(request.form.get("page", 1))
    else:
        anime_name = request.args.get("anime_name", "").strip()
        types_selected = request.args.getlist("types")
        page = int(request.args.get("page", 1))

    if not types_selected:
        types_selected = anime_df["type"].unique().tolist()

    filtered_df = anime_df[anime_df["type"].isin(types_selected)]

    if anime_name:
        mask = (
            filtered_df["name"].str.contains(anime_name, case=False, na=False) |
            filtered_df["genre"].str.contains(anime_name, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        return render_template("recommendations.html", error="No anime matched your criteria.")

    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page

    filtered_indices = filtered_df.index.tolist()

    if anime_name in indices:
        idx = indices[anime_name]
        sim_scores = [(i, cosine_sim[idx][i]) for i in filtered_indices]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [x for x in sim_scores if x[0] != idx]
    else:
        sim_scores = [(i, filtered_df.loc[i, "rating"]) for i in filtered_indices]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores_page = sim_scores[start:end]
    anime_indices_page = [i[0] for i in sim_scores_page]

    recommendations = anime_df.loc[anime_indices_page, ["name", "genre", "type", "rating"]].copy()
    recommendations["score"] = [i[1] for i in sim_scores_page]
    recommendations["image_url"] = recommendations["name"].apply(get_anime_image_url)

    has_next = end < len(sim_scores)
    has_prev = start > 0

    return render_template(
        "recommendations.html",
        recommendations=recommendations.to_dict(orient="records"),
        anime_name=anime_name,
        types_selected=types_selected,
        page=page,
        has_next=has_next,
        has_prev=has_prev
    )

@app.route('/popular', methods=['GET'])
def popular():
    anime_type = request.args.get('type', 'TV')
    page = int(request.args.get('page', 1))
    per_page = 20

    anime_df = pd.read_csv('data/anime.csv')
    ratings_df = pd.read_csv('data/rating.csv')
    ratings_df = ratings_df[ratings_df['rating'] > 0]

    popularity_df = ratings_df.groupby('anime_id').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    merged_df = pd.merge(popularity_df, anime_df, on='anime_id')
    filtered_df = merged_df[merged_df['type'] == anime_type]

    filtered_df['popularity_score'] = filtered_df['rating_count'] * filtered_df['avg_rating']
    sorted_df = filtered_df.sort_values(by='popularity_score', ascending=False)

    total_anime = sorted_df.shape[0]
    start = (page - 1) * per_page
    end = start + per_page

    anime_list = []
    for _, row in sorted_df.iloc[start:end].iterrows():
        anime_list.append({
            'name': row['name'],
            'type': row['type'],
            'avg_rating': row['avg_rating'],
            'rating_count': row['rating_count'],
            'popularity_score': row['popularity_score'],
            'image_url': get_anime_image_url(row['name'])
        })

    has_next = end < total_anime
    has_prev = start > 0

    return render_template(
        'popular.html',
        anime_list=anime_list,
        selected_type=anime_type,
        page=page,
        has_next=has_next,
        has_prev=has_prev
    )

@app.route("/project")
def project():
    return render_template("project.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "development") == "development"
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        threaded=True
    )