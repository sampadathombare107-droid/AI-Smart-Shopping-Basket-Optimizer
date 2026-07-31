import pandas as pd


# ==========================================
# Load Dataset
# ==========================================
def load_data():
    """
    Load cleaned BigBasket dataset.
    """
    return pd.read_csv("data/processed/bigbasket_cleaned.csv")


# ==========================================
# Search Product
# ==========================================
def search_product(df, keyword):
    """
    Search products by keyword.
    """
    if not keyword:
        return pd.DataFrame()

    return df[
        df["product"].str.contains(
            keyword,
            case=False,
            na=False
        )
    ]


# ==========================================
# Get Product Details
# ==========================================
def get_product_details(df, product_name):
    """
    Return first matching product.
    """

    result = search_product(df, product_name)

    if result.empty:
        return None

    return result.iloc[0]


# ==========================================
# Top Rated Products
# ==========================================
def top_rated_products(df, n=10):

    return (
        df.sort_values(
            by="rating",
            ascending=False
        )
        .head(n)
    )


# ==========================================
# Top Brands
# ==========================================
def top_brands(df, n=10):

    return (
        df["brand"]
        .value_counts()
        .head(n)
        .reset_index(name="Products")
        .rename(columns={"brand": "Brand"})
    )


# ==========================================
# Top Categories
# ==========================================
def top_categories(df, n=10):

    return (
        df["category"]
        .value_counts()
        .head(n)
        .reset_index(name="Products")
        .rename(columns={"category": "Category"})
    )


# ==========================================
# Dataset Statistics
# ==========================================
def dataset_statistics(df):

    return {
        "Total Products": len(df),
        "Total Brands": df["brand"].nunique(),
        "Total Categories": df["category"].nunique(),
        "Average Rating": round(df["rating"].mean(), 2)
    }


# ==========================================
# Discount Calculator
# ==========================================
def calculate_discount(sale_price, market_price):

    if market_price == 0:
        return 0

    discount = (
        (market_price - sale_price)
        / market_price
    ) * 100

    return round(discount, 2)
# ==========================================
# Load Association Rules
# ==========================================

def load_association_rules():

    import pandas as pd

    rules = pd.read_csv(
        "data/processed/association_rules.csv"
    )

    return rules