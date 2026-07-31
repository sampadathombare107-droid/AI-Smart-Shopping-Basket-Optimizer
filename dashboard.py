import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show_dashboard(df):

    st.title("📊 Shopping Analytics Dashboard")

    st.markdown("---")

    # -------------------------
    # Dataset Statistics
    # -------------------------

    total_products = len(df)
    total_brands = df["brand"].nunique()
    total_categories = df["category"].nunique()
    average_rating = round(df["rating"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Products", total_products)
    col2.metric("Brands", total_brands)
    col3.metric("Categories", total_categories)
    col4.metric("Average Rating", average_rating)

    st.markdown("---")

    # -------------------------
    # Top Brands
    # -------------------------

    st.subheader("🏷️ Top 10 Brands")

    brand_df = (
        df["brand"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8,5))

    brand_df.plot(kind="bar", ax=ax)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")

    # -------------------------
    # Top Categories
    # -------------------------

    st.subheader("📦 Top Categories")

    category_df = (
        df["category"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8,5))

    category_df.plot(kind="bar", ax=ax)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")

    # -------------------------
    # Rating Distribution
    # -------------------------

    st.subheader("⭐ Rating Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    df["rating"].dropna().plot(
        kind="hist",
        bins=20,
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # -------------------------
    # Sale Price Distribution
    # -------------------------

    st.subheader("💰 Sale Price Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    df["sale_price"].plot(
        kind="hist",
        bins=30,
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    st.success("Dashboard Loaded Successfully")