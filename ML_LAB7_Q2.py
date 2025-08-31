import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# load the dataset and remove lines like @ATTRIBUTE, % etc
file_path = r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\combine.csv"
with open(file_path, "r", errors="ignore") as f:
    lines = f.readlines()
clean_lines = [line for line in lines if not line.strip().startswith(('@', '%'))]

# save the cleaned file
cleaned_path = r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\cleaned.csv"
with open(cleaned_path, "w", encoding="utf-8") as f:
    f.writelines(clean_lines)

# read data into pandas
data = pd.read_csv(cleaned_path, dtype=str, low_memory=False)
data.dropna(how="all", inplace=True)   # drop rows that are fully empty

# separate features and target column
X = data.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
y = data.iloc[:, -1]

# fill missing values in features
X = SimpleImputer(strategy="most_frequent").fit_transform(X)

# encode target labels into numbers
y = LabelEncoder().fit_transform(y.astype(str))

# define the parameter grid for random search
param_dist = {
    'n_estimators': [50, 100],
    'max_depth': [5, None],
    'min_samples_split': [2, 5]
}

# run randomized search with fewer tries and folds to save time
search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=3,
    cv=2,
    random_state=42
)

# train the model
search.fit(X, y)

# print best results
print(" Best Parameters:", search.best_params_)
print(" Best Score:", search.best_score_)
