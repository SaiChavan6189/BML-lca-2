# ============================================================
# BANK MARKETING DATASET
# LOGISTIC REGRESSION
# ============================================================

# -----------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# 2. DOWNLOAD THE DATASET
# -----------------------------

import requests
import zipfile
import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

response = requests.get(url)

print("Dataset downloaded successfully!")


# -----------------------------
# 3. EXTRACT THE DATASET
# -----------------------------

with zipfile.ZipFile(io.BytesIO(response.content)) as z:

    print("\nFiles available in dataset:")
    print(z.namelist())

    with z.open("bank-full.csv") as file:
        data = pd.read_csv(file, sep=";")


# -----------------------------
# 4. DISPLAY DATASET DETAILS
# -----------------------------

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst Five Rows:")
print(data.head())


# -----------------------------
# 5. DATASET INFORMATION
# -----------------------------

print("\nDataset Information:")
print(data.info())


# -----------------------------
# 6. CHECK MISSING VALUES
# -----------------------------

print("\nMissing Values:")
print(data.isnull().sum())


# -----------------------------
# 7. DISPLAY TARGET VALUES
# -----------------------------

print("\nTarget Variable Values:")

print(data["y"].value_counts())


# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n\n========================================")
print("DATA PREPROCESSING")
print("========================================")


# -----------------------------
# 8. SEPARATE INPUT AND OUTPUT
# -----------------------------

X = data.drop("y", axis=1)

y = data["y"].map({
    "no": 0,
    "yes": 1
})


# -----------------------------
# 9. IDENTIFY CATEGORICAL COLUMNS
# -----------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# -----------------------------
# 10. IDENTIFY NUMERICAL COLUMNS
# -----------------------------

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

# -----------------------------
# 11. SPLIT DATA
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# DATA PREPROCESSING PIPELINE
# ============================================================

# -----------------------------
# 12. CREATE PREPROCESSOR
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            numerical_columns
        ),

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

# -----------------------------
# 13. CREATE MODEL
# -----------------------------

model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# -----------------------------
# 14. TRAIN THE MODEL
# -----------------------------

model.fit(
    X_train,
    y_train
)

print("\n========================================")
print("MODEL TRAINING")
print("========================================")

print("\nLogistic Regression model trained successfully!")


# ============================================================
# PREDICTION
# ============================================================

# -----------------------------
# 15. MAKE PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)


print("\nPredictions completed successfully!")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n\n========================================")
print("LOGISTIC REGRESSION RESULTS")
print("========================================")


# -----------------------------
# 16. ACCURACY
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)

print(
    "\nAccuracy Percentage:",
    round(accuracy * 100, 2),
    "%"
)


# -----------------------------
# 17. CONFUSION MATRIX
# -----------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# -----------------------------
# 18. CLASSIFICATION REPORT
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n\n========================================")
print("CONCLUSION")
print("========================================")

print(
    "The Logistic Regression model was successfully "
    "trained and evaluated on the Bank Marketing dataset."
)

print(
    "The model achieved an accuracy of",
    round(accuracy * 100, 2),
    "%"
)
