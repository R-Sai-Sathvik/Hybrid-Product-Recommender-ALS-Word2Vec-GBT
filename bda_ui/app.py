from flask import Flask, jsonify, render_template_string
import pandas as pd
import os
import glob

app = Flask(__name__)

# ─────────────────────────────────────────
# DATA LOADING
# Load once at startup — fast for all requests
# ─────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_parquet(folder):
    files = glob.glob(os.path.join(DATA_DIR, folder, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

print("Loading data... (this takes ~30 seconds on first start)")

print("  Loading recommendations...")
recs_df = load_parquet("hybrid_recommendations")

print("  Loading item index...")
item_index_df = load_parquet("item_index")
# itemIndex is the integer id used in recommendations
asin_lookup = dict(zip(item_index_df["itemIndex"], item_index_df["asin"]))

print("  Loading ratings...")
ratings_df = load_parquet("als_ratings")

print("  Loading item metadata...")
try:
    items_df = load_parquet("bert_items")
    # Build asin → avgRating + reviewCount lookup
    item_meta = items_df[["asin","avgRating","reviewCount"]].drop_duplicates("asin")
    item_meta_lookup = item_meta.set_index("asin").to_dict("index")
except Exception as e:
    print(f"  Warning: Could not load bert_items: {e}")
    item_meta_lookup = {}

print("Data loaded successfully!\n")

# ─────────────────────────────────────────
# PRECOMPUTE STATS (fast dashboard)
# ─────────────────────────────────────────

total_users    = ratings_df["userId"].nunique()
total_items    = ratings_df["itemId"].nunique()
total_ratings  = len(ratings_df)
users_with_recs = recs_df["userId"].nunique()

user_activity = ratings_df.groupby("userId").size().reset_index(name="ratingCount")
cold_users  = int((user_activity["ratingCount"] <= 5).sum())
warm_users  = int((user_activity["ratingCount"] >  5).sum())
power_users = int((user_activity["ratingCount"] >  20).sum())

source_counts = recs_df.groupby("source").agg(
    totalRecs=("itemId","count"),
    usersServed=("userId","nunique")
).reset_index().to_dict("records")

# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────

def enrich_recs(user_recs):
    """Add asin and item metadata to recommendation rows."""
    result = []
    for _, row in user_recs.iterrows():
        item_id  = int(row["itemId"])
        asin     = asin_lookup.get(item_id, f"UNKNOWN_{item_id}")
        meta     = item_meta_lookup.get(asin, {})
        result.append({
            "rank":        int(row["rank"]),
            "itemId":      item_id,
            "asin":        asin,
            "hybridScore": round(float(row["hybridScore"]), 4),
            "alsScore":    round(float(row["alsScore"]),    4),
            "w2vScore":    round(float(row["w2vScore"]),    4),
            "source":      row["source"],
            "avgRating":   round(meta.get("avgRating", 0), 2),
            "reviewCount": int(meta.get("reviewCount", 0)),
            "amazonUrl":   f"https://www.amazon.com/dp/{asin}"
        })
    return result

# ─────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    """Dashboard statistics."""
    return jsonify({
        "totalUsers":     total_users,
        "totalItems":     total_items,
        "totalRatings":   total_ratings,
        "usersWithRecs":  users_with_recs,
        "coveragePct":    round(users_with_recs * 100 / total_users, 2),
        "coldUsers":      cold_users,
        "warmUsers":      warm_users,
        "powerUsers":     power_users,
        "sourceCounts":   source_counts
    })


@app.route("/api/recommend/<int:user_id>")
def api_recommend(user_id):
    """Top-N recommendations for a specific user."""
    from flask import request
    top_n = int(request.args.get("n", 10))
    top_n = max(1, min(top_n, 10))  # clamp between 1 and 10

    user_recs = recs_df[recs_df["userId"] == user_id].sort_values("rank").head(top_n)

    if user_recs.empty:
        # Check if user exists at all
        user_exists = user_id in ratings_df["userId"].values
        return jsonify({
            "userId":    user_id,
            "found":     False,
            "userExists": bool(user_exists),
            "message":   "No recommendations found for this user.",
            "recs":      []
        })

    user_info = user_activity[user_activity["userId"] == user_id]
    rating_count = int(user_info["ratingCount"].values[0]) if len(user_info) > 0 else 0

    segment = "cold-start"
    if rating_count > 20:
        segment = "power user"
    elif rating_count > 5:
        segment = "regular user"

    source = user_recs["source"].iloc[0]

    return jsonify({
        "userId":      user_id,
        "found":       True,
        "ratingCount": rating_count,
        "segment":     segment,
        "source":      source,
        "recs":        enrich_recs(user_recs)
    })


@app.route("/api/search_users")
def api_search_users():
    """Return sample user IDs — ONLY from users who actually have recommendations."""
    # Get users that have recommendations and merge with their rating count
    users_with_recs = recs_df[["userId"]].drop_duplicates()
    users_with_recs = users_with_recs.merge(user_activity, on="userId", how="left")

    cold_rec_users  = users_with_recs[users_with_recs["ratingCount"] <= 5]
    warm_rec_users  = users_with_recs[(users_with_recs["ratingCount"] > 5) & (users_with_recs["ratingCount"] <= 20)]
    power_rec_users = users_with_recs[users_with_recs["ratingCount"] > 20]

    sample_cold  = cold_rec_users["userId"].sample(min(5, len(cold_rec_users))).tolist()
    sample_warm  = warm_rec_users["userId"].sample(min(5, len(warm_rec_users))).tolist()
    sample_power = power_rec_users["userId"].sample(min(5, len(power_rec_users))).tolist()

    return jsonify({
        "coldUsers":  [int(x) for x in sample_cold],
        "warmUsers":  [int(x) for x in sample_warm],
        "powerUsers": [int(x) for x in sample_power]
    })


@app.route("/api/model_metrics")
def api_model_metrics():
    """GBT model metrics — hardcoded from last training run."""
    return jsonify({
        "aucROC":   0.8245,
        "aucPR":    0.4251,
        "accuracy": 0.9037,
        "featureImportances": [
            {"feature": "logUserActivity", "importance": 0.2470},
            {"feature": "logItemPop",      "importance": 0.2292},
            {"feature": "itemAvgRating",   "importance": 0.1852},
            {"feature": "w2vScore",        "importance": 0.1836},
            {"feature": "alsScore",        "importance": 0.1551}
        ],
        "trainedOn": "2026-03-03",
        "model":     "GBT Classifier (maxIter=50, maxDepth=5)"
    })


# ─────────────────────────────────────────
# FRONTEND — served from Flask directly
# ─────────────────────────────────────────

@app.route("/")
def index():
   return render_template_string(open(
        os.path.join(os.path.dirname(__file__), "index.html"),
        encoding="utf-8"
    ).read())


if __name__ == "__main__":
    print("Starting BDA Recommender UI at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)