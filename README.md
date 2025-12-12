#📦 Automatic Classification of Products by Category
Machine Learning project developed as part of the IT Academy program
This project focuses on building an automated system that predicts the product category based on its title.
It is developed as part of my Machine Learning studies at IT Akademija, with the goal of understanding the complete ML workflow — from raw data to a deployed prediction script.

The pipeline includes:

Data cleaning & preprocessing
Category normalization and keyword extraction
Feature engineering (TF-IDF + custom binary features)
Model training and evaluation
Saving the trained model
Creating an interactive prediction script
## 🚀 Project Structure
project/
│
├── data/
│   └── products.csv
│
├── model/
│   ├── product_category_model.pkl       # trained ML pipeline
│   ├── category_words.pkl               # extracted category-specific keywords
│   └── binary_columns.pkl               # list of engineered binary feature columns
│
├── src/
│   ├── train_model.py                   # trains the model and saves artifacts
│   └── test_model.py                    # interactive script for predictions
│
└── README.md
### 🧹 1. Data Cleaning & Preprocessing
The dataset contains product titles with inconsistent formatting, duplicate values, and irregular category names.

The preprocessing steps include:

Cleaning column names
Lowercasing and stripping product titles
Normalizing category labels (e.g., "fridge", "fridges" → "fridge freezers")
Removing duplicates
Extracting the most representative keywords per category
These keywords are later used to create additional binary feature columns (has_tvs_word, has_cpus_word, etc.).

### 🛠 2. Feature Engineering
Two types of features are used:

🔹 TF-IDF features
Transform the product title text into numerical vectors.

🔹 Binary category-specific features
For every category, a set of unique keywords is extracted. Example:

mobile phones → ["sim", "gb", "dual", "iphone", "samsung"]
From this, binary columns are generated:

has_mobile_phones_word
has_tvs_word
has_microwaves_word
...
If a product title contains any keyword for a category → value = 1, else 0.

These binary features dramatically improve model accuracy.

### 🤖 3. Model Training
train_model.py performs the full training pipeline:

Loads and cleans the dataset
Generates category-specific keywords
Creates binary feature columns
Splits the data (train/test)
Builds a ColumnTransformer combining TF-IDF and binary features
Trains an SVM classifier (best-performing)
Saves:
Trained model pipeline
Extracted keywords
Binary feature column names
The final model achieves 97–99% accuracy across categories.

### 🎯 4. Interactive Prediction Script
test_model.py allows you to enter a product title in the terminal:

Enter a product name: iphone 7 32gb gold,4,3,Apple iPhone 7 32GB
Predicted category: mobile phones
The script:

Loads the trained model
Loads saved category keywords and binary feature names
Recreates all features for the user-provided text
Outputs the predicted category
Works fully offline once the model is trained.

▶ How to Run
1. Train the model (optional):
python src/train_model.py
2. Run the prediction script:
python src/test_model.py
Then simply type product names interactively.

📚 Technologies Used
Python
Pandas
Scikit-learn
Joblib
Regular Expressions (Regex)
🎓 Educational Purpose
This project is created as part of my Machine Learning studies at the IT Academy, where I am practicing end-to-end ML workflows, feature engineering, model building, and deployment of simple ML applications.

💬 Questions or improvements?
I am actively learning — feel free to open an Issue or suggest improvements!