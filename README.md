---

📦 Automatic Classification of Products by Category

Machine Learning Project (IT Academy)

This is a Machine Learning project developed as part of my studies at IT Academy.
The goal of the project is to build an automated system that predicts a product category based only on its title.

This project helped me practice the end-to-end ML workflow:

Data cleaning & preprocessing

Category normalization

Keyword extraction

Feature engineering (TF-IDF + custom binary features)

Model training & evaluation

Saving the trained model

Creating an interactive prediction script



---

📁 Project Structure

project/
│
├── data/
│ └── products.csv
│
├── model/
│ ├── product_category_model.pkl # trained ML pipeline
│ ├── category_words.pkl # extracted category-specific keywords
│ └── binary_columns.pkl # list of engineered binary feature columns
│
├── src/
│ ├── train_model.py # trains model and saves artifacts
│ └── test_model.py # interactive prediction script
│
└── README.md


---

1️⃣ Data Cleaning & Preprocessing

The dataset contains product titles with inconsistent formatting, duplicate values, and irregular category names.

Preprocessing steps:

Lowercasing and stripping product titles

Normalizing category labels (e.g., "fridge", "fridges", "fridge freezers")

Removing duplicates

Extracting the most representative keywords per category


These keywords are later used to create additional binary feature columns such as:

has_tvs_word

has_cpus_word

has_microwaves_word

has_mobile_phones_word


Each feature = 1 if a product title contains a keyword for that category, otherwise 0.

These binary features significantly improve model accuracy.


---

2️⃣ Feature Engineering

Two feature types were used:

🔹 TF-IDF features

Convert product title text into numerical vectors.

🔹 Category-specific binary features

Created using the extracted keywords for each category.

Example keywords for mobile phones:
["sim", "gb", "dual", "iphone", "samsung"]

From this, the following features are generated:

has_mobile_phones_word

has_tvs_word

has_cpus_word

has_microwaves_word



---

3️⃣ Model Training

train_model.py performs the full training pipeline:

Loads and cleans the dataset

Generates category-specific keywords

Creates binary feature columns

Splits the data

Builds a ColumnTransformer combining TF-IDF and binary features

Trains a model (SVM performs best)

Saves:


product_category_model.pkl
category_words.pkl
binary_columns.pkl

The final model achieves 97–99% accuracy across categories.


---

4️⃣ Interactive Prediction Script

test_model.py allows you to enter a product title in the terminal and receive a predicted category.

Example:

Enter a product name: iphone 7 32gb gold,4,3,Apple iPhone 7 32GB  
Predicted category: mobile phones

The script:

Loads the trained model

Loads category keywords and binary features

Recreates all features from the user-entered text

Outputs the predicted category

Works offline once the model is trained



---

▶️ How to Run

1. Train the model (optional if you already have .pkl files)

python src/train_model.py

2. Run the prediction script

python src/test_model.py

Then simply type product names interactively.


---

🛠 Technologies Used

Python

Pandas

Scikit-learn

Joblib

Regular Expressions (Regex)



---

🎓 Educational Purpose

This project is created as part of my Machine Learning studies at IT Akademija, where I am practicing:

End-to-end ML workflows

Feature engineering

Model building and evaluation

Deploying simple ML applications



---

❓ Questions or Improvements?

I am actively learning — feel free to open an Issue or suggest improvements!


---

