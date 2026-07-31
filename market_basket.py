import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/raw/online_retail/online_retail_II.csv")

print("Dataset Loaded Successfully")
print(df.shape)

# ==========================================
# Data Cleaning
# ==========================================

df = df.dropna(subset=["Invoice", "Description"])

# Remove cancelled invoices
df = df[~df["Invoice"].astype(str).str.startswith("C")]

# Keep only positive quantities
df = df[df["Quantity"] > 0]

print("After Cleaning :", df.shape)
# ==========================================
# Filter Dataset (Reduce Memory Usage)
# ==========================================

# Keep only United Kingdom transactions
df = df[df["Country"] == "United Kingdom"]

# Keep only popular products (appearing at least 100 times)
product_counts = df["Description"].value_counts()

popular_products = product_counts[product_counts >= 100].index

df = df[df["Description"].isin(popular_products)]

print("After Filtering :", df.shape)

# ==========================================
# Basket Creation
# ==========================================

basket = (
    df.groupby(["Invoice", "Description"])["Quantity"]
      .sum()
      .unstack()
      .fillna(0)
)

basket = (basket > 0)
print("Basket Shape :", basket.shape)

# ==========================================
# Apriori
# ==========================================

print("Running Apriori...")

frequent_items = apriori(
    basket,
    min_support=0.03,
    use_colnames=True
)

print("Frequent Itemsets :", len(frequent_items))

# ==========================================
# Association Rules
# ==========================================

rules = association_rules(
    frequent_items,
    metric="lift",
    min_threshold=1
)

rules = rules.sort_values(
    by="lift",
    ascending=False
)

print("Rules Generated :", len(rules))

# ==========================================
# Save Rules
# ==========================================

rules.to_csv(
    "data/processed/association_rules.csv",
    index=False
)

print("===================================")
print("association_rules.csv saved!")
print("===================================")

print(rules.head())