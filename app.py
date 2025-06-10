from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the datasets
anime_df = pd.read_csv("data/anime.csv")
rating_df = pd.read_csv("data/rating.csv")

# Clean and prepare data
anime_df.dropna(subset=["genre", "type"], inplace=True)
anime_df["genre"] = anime_df["genre"].fillna("Unknown")
anime_df["rating"] = anime_df["rating"].fillna(anime_df["rating"].median())
rating_df = rating_df[rating_df["rating"] != -1]

anime_df["combined_features"] = anime_df["genre"] + " " + anime_df["type"] + " " + anime_df["name"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(anime_df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(anime_df.index, index=anime_df["name"]).drop_duplicates()

def get_anime_image_url(title):
    try:
        url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
        response = requests.get(url)
        data = response.json()
        if data["data"]:
            return data["data"][0]["images"]["jpg"]["image_url"]
    except Exception as e:
        print(f"Image fetch failed for {title}: {e}")
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
    app.run(debug=True)