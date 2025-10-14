import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_tabular import LimeTabularExplainer
import shap

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def load_dataset(path):
    """Load dataset, clean bad characters, detect target column."""
    df = pd.read_csv(path, low_memory=False)
    print(f" Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean control characters like \x1a
    df.replace(to_replace=r'[\x00-\x1f\x7f-\x9f]', value=np.nan, regex=True, inplace=True)
    df.dropna(how="all", inplace=True)

    # Detect target column
    possible_targets = ['label', 'target', 'class', 'class1', 'category', 'output', 'author']
    target_col = None
    for col in df.columns:
        if col.lower() in possible_targets:
            target_col = col
            break
    if target_col is None:
        raise ValueError(" Could not detect target column. Rename it to 'label' or 'target'.")

    print(f" Target column detected: '{target_col}'")

    # Encode target if it’s categorical
    if df[target_col].dtype == 'object':
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    return df, target_col


def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numeric features (A1)."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Heatmap (A1)")
    plt.show()


def preprocess_features(df, target):
    """Convert all features to numeric, replacing invalid values."""
    X = df.drop(columns=[target])
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, invalid entries → NaN
    X = X.fillna(0)  # Replace NaN with 0
    return X


def run_pca(X_train, X_test, variance_ratio):
    """Perform PCA retaining given explained variance (A2/A3)."""
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    
    pca = PCA(n_components=variance_ratio)
    X_pca_train = pca.fit_transform(X_scaled_train)
    X_pca_test = pca.transform(X_scaled_test)
    print(f" PCA retained {pca.n_components_} components for {variance_ratio*100:.0f}% variance")
    return X_pca_train, X_pca_test, pca


def evaluate_model(model, X_train, X_test, y_train, y_test, label="Model"):
    """Train and evaluate model performance."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n {label} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    return acc


def run_fast_sequential_feature_selection(model, X_train, y_train, n_features=10):
    """
    Faster version of Sequential Feature Selection (A4)
    - Samples 5000 rows max
    - Uses all CPU cores
    """
    print(" Running faster Sequential Feature Selection (A4)...")
    sample_size = min(5000, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    y_sample = y_train.loc[X_sample.index]

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction='forward',
        n_jobs=-1
    )
    sfs.fit(X_sample, y_sample)
    selected_idx = sfs.get_support(indices=True)
    print(f"🔍 Selected feature indices: {selected_idx}")
    return selected_idx, sfs


def explain_with_lime(model, X_train, X_test, feature_names):
    """Explain a sample prediction using LIME (A5) - script-friendly."""
    print(" Generating LIME explanation (A5)...")
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[f"Class {i}" for i in np.unique(model.predict(X_train))],
        mode="classification"
    )
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=5)
    
    # Script-friendly output
    print("\nTop features affecting prediction for the first test instance:")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")


def explain_with_shap(model, X_train, X_test):
    """Explain model behavior using SHAP (A5)."""
    print(" Generating SHAP explanation (A5)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=True)


# ------------------------------------------------------------
# Main Program
# ------------------------------------------------------------

def main():
    # === Load Dataset ===
    df, target = load_dataset(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\cleaned.csv")

    # === A1: Feature Correlation Analysis ===
    plot_correlation_heatmap(df)

    # === Prepare Data ===
    X = preprocess_features(df, target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_model = RandomForestClassifier(random_state=42)

    # === A2: PCA (99% Variance) ===
    X_pca_train_99, X_pca_test_99, _ = run_pca(X_train, X_test, 0.99)
    acc_99 = evaluate_model(base_model, X_pca_train_99, X_pca_test_99, y_train, y_test, "PCA 99%")

    # === A3: PCA (95% Variance) ===
    X_pca_train_95, X_pca_test_95, _ = run_pca(X_train, X_test, 0.95)
    acc_95 = evaluate_model(base_model, X_pca_train_95, X_pca_test_95, y_train, y_test, "PCA 95%")

    # === A4: Fast Sequential Feature Selection ===
    selected_idx, _ = run_fast_sequential_feature_selection(base_model, X_train, y_train, n_features=min(10, X_train.shape[1]))
    X_train_sfs = X_train.iloc[:, selected_idx]
    X_test_sfs = X_test.iloc[:, selected_idx]
    selected_features = X_train_sfs.columns.tolist()
    acc_sfs = evaluate_model(base_model, X_train_sfs, X_test_sfs, y_train, y_test, "Sequential FS (Fast)")

    # === A5: Explainability with LIME and SHAP ===
    explain_with_lime(base_model, X_train_sfs.values, X_test_sfs.values, selected_features)
    explain_with_shap(base_model, X_train_sfs, X_test_sfs)

    # === Summary ===
    print("\n============== RESULT SUMMARY ==============")
    print(f"PCA 99% Accuracy: {acc_99:.4f}")
    print(f"PCA 95% Accuracy: {acc_95:.4f}")
    print(f"Sequential FS (Fast) Accuracy: {acc_sfs:.4f}")
    print("============================================")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()