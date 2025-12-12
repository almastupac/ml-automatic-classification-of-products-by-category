import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# ---------------------------------------------------------
# 1. Load and clean the dataset
# ---------------------------------------------------------

df = pd.read_csv("data/products.csv")

# Drop missing values
df = df.dropna()

# Clean column names
df.columns = (
    df.columns
    .str.strip()  # remove leading/trailing spaces
    .str.lower()  # lowercase
    .str.lstrip("_")  # remove leading underscore
    .str.replace(" ", "_")  # replace spaces
    .str.replace("-", "_")  # replace hyphens
    .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)  # remove special characters
)

# Clean product_title text
#  Convert 'product_title' column to string and lowercase
df['product_title'] = df['product_title'].astype(str).str.lower()

# Remove multiple spaces
import re
df['product_title'] = df['product_title'].str.replace(r"\s+", " ", regex=True).str.strip()

# Standardize number formats and memory labels
# Example: unify formats like "16 gb", "16gb", "16/gb" into "16gb"
df['product_title'] = df['product_title'].str.replace(r"(\d+)\s*gb", r"\1gb", regex=True)

# Removing special characters that are not informative
df['product_title'] = df['product_title'].str.replace(r"[^a-z0-9 ]", " ", regex=True)

# Remove duplicate entries after all cleaning steps
df = df.drop_duplicates(subset=['product_title'])

# Standardize category names
df["category_label"] = (
    df["category_label"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# Standardization dictionary for category_label
category_map = {
    # Mobile phones
    'mobile phone': 'mobile phones',
    'mobile phones': 'mobile phones',
    'phone': 'mobile phones',
    'phones': 'mobile phones',
    'smartphones': 'mobile phones',

    # Fridges / freezers
    'fridge freezer': 'fridge freezers',
    'fridges': 'fridge freezers',
    'freezer': 'fridge freezers',
    'freezers': 'fridge freezers',
    'fridges': 'fridge freezers',
    'fridge': 'fridge freezers',

    # Washing machines
    'washing machine': 'washing machines',
    'washing machines': 'washing machines',

    # CPUs
    'cpu': 'cpus',
    'cpus': 'cpus',

    # TVs
    'tv': 'tvs',
    'tvs': 'tvs',

    # Cameras
    'digital camera': 'digital cameras',
    'digital cameras': 'digital cameras',

    # Microwaves
    'microwave': 'microwaves',
    'microwaves': 'microwaves'

}

# Convert to lowercase first
df['category_label'] = df['category_label'].astype(str).str.lower().str.strip()

# Apply mapping
df['category_label'] = df['category_label'].replace(category_map)

# Drop columns that are not useful for modeling
df = df.drop(columns=['product_id', 'merchant_id', 'product_code', 'number_of_views', 'merchant_rating', 'listing_date'])
# Get all unique catgory names
categories = df['category_label'].unique().tolist()

# Compute category-specific keywords
category_specific_words = {}

# Minimum ferquency inside the category
min_cat_count = 20

# Maximum allowes ratio of occurrences in other categories
max_other_ratio = 0.2

for cat in categories:
  # Titles within the current category
  titles_cat = df[df['category_label'] == cat]['product_title']
  words_cat = titles_cat.str.split().explode().value_counts()

  # Titles from all other categories
  titles_other = df[df['category_label'] != cat]['product_title']
  words_other = titles_other.str.split().explode().value_counts()

  specific = []

  # Loop through all words in the category
  for word, cnt in words_cat.items():
    other_cnt = words_other.get(word, 0)
    ratio = other_cnt / (cnt + 1)

    # Filter: ferquent in category, rare in others, alphabetic, length > 2
    if (
        cnt >= min_cat_count and
        ratio < max_other_ratio and
        word.isalpha()  and
        len(word) > 2
    ):specific.append(word)

  # Keep top 30 strongest keywords for this category
  category_specific_words[cat] = specific[:30]

# Creating binary feature columns based on category-specific keywords

for cat, words in category_specific_words.items():
  # Create a column name such as "has_mobile_phones_word"
  col_name = f"has_{cat.replace(' ', '_')}_word"

  # Assign 1 if any keyword for this category appears in the product title
  df[col_name] = df['product_title'].apply(lambda text: int(any(word in text.split() for word in words)))

# Features and labels
# All binary feature columns from feature engineering
binary_features = [col for col in df.columns if col.startswith("has_")]

# Combine text + binary columns into X
X = df[['product_title'] + binary_features]

# Target label
y = df['category_label']

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "product_title"),
        ("binary", "passthrough", binary_features)
    ]
)

# Define pipeline with best model (e.g.SVC)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, "model/product_category_model.pkl")
print("\nModel saved as model/product_category_model.pkl")