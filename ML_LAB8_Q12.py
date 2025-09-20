# ML_LAB8_Q12_robust.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

# --------------------------
# Load dataset safely
# --------------------------
data = pd.read_csv(
    r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\cleaned.csv",
    dtype=str,
    low_memory=False,
    on_bad_lines="skip"
)

# Drop completely empty rows
data = data.dropna(how="all")

# --------------------------
# Separate features and label
# --------------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode labels if not numeric
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

# --------------------------
# Remove rare classes (only 1 sample) for stratification
# --------------------------
class_counts = Counter(y)
valid_classes = [cls for cls, count in class_counts.items() if count > 1]
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]

# --------------------------
# Preprocess Features
# --------------------------

# Numeric columns
numeric_cols = X.columns[X.apply(lambda col: pd.to_numeric(col, errors="coerce").notnull().all())]
X_numeric = X[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Categorical columns
categorical_cols = X.select_dtypes(include='object').columns

# Low-cardinality → one-hot encode
low_card_cols = [col for col in categorical_cols if X[col].nunique() <= 20]
if low_card_cols:
    X_low_card = pd.get_dummies(X[low_card_cols], drop_first=True)
else:
    X_low_card = pd.DataFrame()

# High-cardinality → label encode
high_card_cols = [col for col in categorical_cols if X[col].nunique() > 20]
if high_card_cols:
    X_high_card = X[high_card_cols].apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
else:
    X_high_card = pd.DataFrame()

# Combine all features
X = pd.concat([X_numeric, X_low_card, X_high_card], axis=1)

# --------------------------
# Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# Train MLP
# --------------------------
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

# --------------------------
# Results
# --------------------------
print("Training complete")
print("Training accuracy:", mlp.score(X_train, y_train))
print("Testing accuracy:", mlp.score(X_test, y_test))
