import joblib
import pandas as pd
import re

# -------------------------------------------------------------------
# Load the saved model and the helper objects created during training
# -------------------------------------------------------------------
model = joblib.load("model/product_category_model.pkl") # trained Pipeline
category_specific_words = joblib.load("model/category_words.pkl") # dict: category → list of keywords
binary_columns = joblib.load("model/binary_columns.pkl") # list of all has_* feature columns

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")


# -------------------------------------------------------------------
# Function: create all required input features for prediction
# -------------------------------------------------------------------
def make_features(title: str) -> pd.DataFrame:
    """
    Creates a one-row DataFrame containing:
    - product_title (raw input text)
    - all binary has_*_word columns expected by the model
    """

    # Normalize product title
    text = str(title).lower().strip()
    tokens = re.findall(r"\w+", text)

    # Start row with product_title
    row = {"product_title": text}

    # Add every binary feature column the model expects
    for col in binary_columns:

        # Convert column name back to readable category name
        # Example: has_mobile_phones_word → mobile phones
        cat = (
            col.replace("has_", "")
               .replace("_word", "")
               .replace("_", " ")
        )

        # Retrieve keyword list for this category
        words = category_specific_words.get(cat, [])

        # Assign 1 if any keyword is found in the user input
        row[col] = int(any(w in tokens for w in words))

    # Return as one-row DataFrame
    return pd.DataFrame([row])


# -------------------------------------------------------------------
# Interactive prediction loop
# -------------------------------------------------------------------
while True:
    title = input("Enter a product name: ")

    if title.lower() == "exit":
        print("Exiting...")
        break

    # Create full feature set for the model
    X_input = make_features(title)

    # Predict category
    prediction = model.predict(X_input)[0]

    print(f"Predicted category: {prediction}")
    print("-" * 40)