import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Load Dataset
# ==========================================

def load_products():
    return pd.read_csv("data/processed/bigbasket_cleaned.csv")


# ==========================================
# Prepare Text Features
# ==========================================

def prepare_text(df):

    df = df.copy()

    df["combined_features"] = (
        df["product"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["sub_category"].fillna("") + " " +
        df["brand"].fillna("") + " " +
        df["description"].fillna("")
    )

    return df


# ==========================================
# Build TF-IDF Model
# ==========================================

def build_model(df):

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        df["combined_features"]
    )

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    return similarity_matrix


# ==========================================
# Recommend Similar Products
# ==========================================

def recommend_products(
    product_name,
    df,
    similarity_matrix,
    top_n=5
):

    matched = df[
        df["product"].str.contains(
            product_name,
            case=False,
            na=False
        )
    ]

    if matched.empty:
        return pd.DataFrame()

    index = matched.index[0]

    scores = list(
        enumerate(similarity_matrix[index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    indices = [
        item[0]
        for item in scores[1:top_n + 1]
    ]

    recommendations = df.iloc[indices].copy()

    recommendations["Similarity Score"] = [
        round(item[1] * 100, 2)
        for item in scores[1:top_n + 1]
    ]

    return recommendations[
        [
            "product",
            "brand",
            "category",
            "sale_price",
            "market_price",
            "rating",
            "Similarity Score"
        ]
    ]


# ==========================================
# Complete Recommendation Pipeline
# ==========================================

def get_recommendations(product_name):

    df = load_products()

    df = prepare_text(df)

    similarity_matrix = build_model(df)

    return recommend_products(
        product_name,
        df,
        similarity_matrix
    )