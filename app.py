import streamlit as st
from utils import load_association_rules
from utils import (
    load_data,
    dataset_statistics,
    search_product,
    get_product_details,
    calculate_discount,
    load_association_rules
)
from recommender import (
    prepare_text,
    build_model,
    recommend_products
)

from dashboard import show_dashboard

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Smart Shopping Basket Optimizer",
    page_icon="🛒",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def get_data():
    return load_data()

df = get_data()

stats = dataset_statistics(df)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🛒 AI Shopping Optimizer")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🔍 Product Search",
        "🤖 AI Recommendation",
        "🛒 Frequently Bought Together",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🛒 AI Smart Shopping Basket Optimizer")

    st.caption("Machine Learning Based Recommendation System")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Products",
        stats["Total Products"]
    )

    c2.metric(
        "Brands",
        stats["Total Brands"]
    )

    c3.metric(
        "Categories",
        stats["Total Categories"]
    )

    c4.metric(
        "Avg Rating",
        stats["Average Rating"]
    )

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
The objective of this project is to build an intelligent shopping
assistant that recommends products using Machine Learning.

The application performs

• Product Search

• Shopping Analytics

• Similar Product Recommendation

• Market Basket Analysis
""")

    st.markdown("---")

    st.header("🤖 Machine Learning Algorithms")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("1. Apriori Algorithm")

        st.write("""
Purpose

• Market Basket Analysis

• Frequently Bought Together

Metrics

• Support

• Confidence

• Lift
""")

    with col2:

        st.subheader("2. TF-IDF + Cosine Similarity")

        st.write("""
Purpose

• Content Based Recommendation

• Similar Product Detection

Technique

• TF-IDF Vectorization

• Cosine Similarity
""")

    st.markdown("---")

    st.header("📂 Dataset")

    st.write(f"Rows : {len(df)}")

    st.write(f"Columns : {len(df.columns)}")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

# ==========================================
# DASHBOARD
# ==========================================

elif page == "📊 Dashboard":

    show_dashboard(df)

# ==========================================
# PRODUCT SEARCH
# ==========================================

# ==========================================
# PRODUCT SEARCH
# ==========================================

elif page == "🔍 Product Search":

    st.title("🔍 Product Search")

    st.write("Search products available in the BigBasket dataset.")

    keyword = st.text_input(
        "Enter Product Name"
    )

    if keyword:

        results = search_product(df, keyword)

        if results.empty:

            st.error("No product found.")

        else:

            st.success(
                f"{len(results)} Product(s) Found"
            )

            st.dataframe(
                results[
                    [
                        "product",
                        "brand",
                        "category",
                        "sale_price",
                        "market_price",
                        "rating"
                    ]
                ],
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("📦 Product Details")

            product = get_product_details(
                df,
                keyword
            )

            if product is not None:

                discount = calculate_discount(
                    product["sale_price"],
                    product["market_price"]
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Sale Price",
                        f"₹{product['sale_price']}"
                    )

                    st.metric(
                        "Market Price",
                        f"₹{product['market_price']}"
                    )

                    st.metric(
                        "Discount",
                        f"{discount}%"
                    )

                with col2:

                    st.metric(
                        "Rating",
                        product["rating"]
                    )

                    st.write(
                        f"**Brand:** {product['brand']}"
                    )

                    st.write(
                        f"**Category:** {product['category']}"
                    )

                    st.write(
                        f"**Sub Category:** {product['sub_category']}"
                    )

                st.markdown("### Description")

                st.write(
                    product["description"]
                )

                st.markdown("---")

                st.subheader("⭐ Top Rated Matching Products")

                top_products = results.sort_values(
                    by="rating",
                    ascending=False
                ).head(5)

                st.dataframe(
                    top_products[
                        [
                            "product",
                            "brand",
                            "sale_price",
                            "rating"
                        ]
                    ],
                    use_container_width=True
                )

# ==========================================
# AI RECOMMENDATION
# ==========================================

elif page == "🤖 AI Recommendation":

    st.title("🤖 AI Product Recommendation")

    st.write(
        "Find products similar to your selected product using Machine Learning."
    )

    search = st.text_input(
        "Enter Product Name",
        key="recommend"
    )

    if search:

        model_df = prepare_text(df)

        similarity_matrix = build_model(model_df)

        recommendations = recommend_products(
            search,
            model_df,
            similarity_matrix
        )

        if recommendations.empty:

            st.error("No similar products found.")

        else:

            st.success("Top Similar Products")

            st.dataframe(
                recommendations,
                use_container_width=True
            )

# ==========================================
# FREQUENTLY BOUGHT TOGETHER
# ==========================================

elif page == "🛒 Frequently Bought Together":

    st.title("🛒 Frequently Bought Together")

    st.write(
        "Products frequently purchased together using Apriori Algorithm."
    )

    rules = load_association_rules()

    st.success(f"{len(rules)} Association Rules Loaded")

    display_rules = rules.copy()

    display_rules["antecedents"] = (
        display_rules["antecedents"]
        .astype(str)
        .str.replace("frozenset({", "", regex=False)
        .str.replace("})", "", regex=False)
        .str.replace("'", "", regex=False)
    )

    display_rules["consequents"] = (
        display_rules["consequents"]
        .astype(str)
        .str.replace("frozenset({", "", regex=False)
        .str.replace("})", "", regex=False)
        .str.replace("'", "", regex=False)
    )

    display_rules["support"] = (
        display_rules["support"] * 100
    ).round(2)

    display_rules["confidence"] = (
        display_rules["confidence"] * 100
    ).round(2)

    display_rules["lift"] = (
        display_rules["lift"]
    ).round(2)

    display_rules = display_rules.rename(
        columns={
            "antecedents": "Bought Product",
            "consequents": "Recommended Product",
            "support": "Support (%)",
            "confidence": "Confidence (%)",
            "lift": "Lift Score"
        }
    )

    st.dataframe(
        display_rules[
            [
                "Bought Product",
                "Recommended Product",
                "Support (%)",
                "Confidence (%)",
                "Lift Score"
            ]
        ],
        use_container_width=True
    )

# ==========================================
# ABOUT
# ==========================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""
# 🛒 AI Smart Shopping Basket Optimizer

## Machine Learning Project

### Technologies Used

- Python
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Scikit-learn
- MLxtend

### Machine Learning Algorithms

- TF-IDF Vectorization
- Cosine Similarity
- Apriori Algorithm
- Association Rule Mining

### Features

- 🔍 Product Search
- 📊 Shopping Dashboard
- 🤖 AI Product Recommendation
- 🛒 Frequently Bought Together
- 📈 Shopping Analytics

### Dataset

- BigBasket Product Dataset
- Online Retail II Dataset

### Developed By

**Sampada Thombare**
""")