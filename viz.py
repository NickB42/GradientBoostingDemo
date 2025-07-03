"""
Bank-Marketing – Gradient-Boosting Demo (erweitert) mit Visualisierungen
------------------------------------------------------------------------
• Datensatz: UCI Bank Marketing (45 211 Zeilen, 17 Features, Binärziel «y»)
• Modelle : Logistic Regression (Baseline), Decision Tree, Random Forest,
            GradientBoostingClassifier, XGBoostClassifier
• Metriken: Accuracy, F1, ROC-AUC
• Visualisierungen: ROC-Kurven, Confusion Matrix, Feature Importance, etc.
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from collections import Counter
from sklearn.tree import plot_tree  

def visualize_trees(models, feature_names, max_depth_display=3):
    """
    Visualisiert Decision Trees und Random Forest Komponenten
    """
    
    # =============================================================================
    # 1. DECISION TREE VOLLSTÄNDIGE VISUALISIERUNG
    # =============================================================================
    
    if "Decision Tree" in models:
        dt_model = models["Decision Tree"]
        
        # Kompletter Baum (falls nicht zu groß)
        plt.figure(figsize=(20, 12))
        plot_tree(dt_model, 
                 feature_names=feature_names,
                 class_names=['No', 'Yes'],
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 max_depth=max_depth_display)  # Begrenzt Tiefe für Lesbarkeit
        plt.title(f'Decision Tree (max depth display: {max_depth_display})', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # =============================================================================
    # 2. RANDOM FOREST - EINZELNE BÄUME VISUALISIEREN
    # =============================================================================
    
    if "Random Forest" in models:
        rf_model = models["Random Forest"]
        
        # Visualisiere erste 3 Bäume aus dem Random Forest
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        for i in range(3):
            tree = rf_model.estimators_[i]
            plot_tree(tree,
                     feature_names=feature_names,
                     class_names=['No', 'Yes'],
                     filled=True,
                     rounded=True,
                     fontsize=6,
                     max_depth=3,  # Begrenzt für Lesbarkeit
                     ax=axes[i])
            axes[i].set_title(f'Random Forest - Baum {i+1}', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def visualize_boosting_trees(models, feature_names, n_trees=3, rankdir="LR"):
    """
    Show the first `n_trees` of GradientBoostingClassifier and XGBClassifier models.
    -------------------------------------------------------------------------------
    • models         – dict like yours: {"GradientBoosting": model, "XGBoost": model, …}
    • feature_names  – list of column names after preprocessing (same one you already build)
    • n_trees        – how many boosting stages to draw
    • rankdir        – "TB" (top-bottom) or "LR" (left-right) layout for XGBoost
    """

    # ------------------------------------------------------------------
    # 1) sklearn GradientBoostingClassifier
    # ------------------------------------------------------------------
    if "GradientBoosting" in models:
        gbrt = models["GradientBoosting"]
        # gbrt.estimators_  → shape (n_estimators, n_classes)  (binary ⇒ [:, 0])
        fig, axes = plt.subplots(1, n_trees, figsize=(5 * n_trees, 5))
        for i in range(n_trees):
            stump = gbrt.estimators_[i, 0]          # zero-indexed stage
            plot_tree(
                stump,
                feature_names=feature_names,
                class_names=["No", "Yes"],
                filled=True,
                rounded=True,
                fontsize=7,
                ax=axes[i]
            )
            axes[i].set_title(f"GB Stage {i+1}", weight="bold")
        plt.tight_layout()
        plt.show()

# Styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

RANDOM_STATE = 42

# Datensatz laden (wie vorher)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
LOCAL_ZIP = "bank.zip"
CSV_PATH = "bank-full.csv"

if not os.path.exists(CSV_PATH):
    print("➜ Lade Datensatz …")
    urllib.request.urlretrieve(DATA_URL, LOCAL_ZIP)
    with zipfile.ZipFile(LOCAL_ZIP) as zf:
        zf.extract(CSV_PATH)
    os.remove(LOCAL_ZIP)

# Daten laden und vorbereiten
df = pd.read_csv(CSV_PATH, sep=';')
print("Shape:", df.shape)

X = df.drop("y", axis=1)
y = df["y"].map({"yes": 1, "no": 0}).astype(np.int8)

cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ]
)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Daten vorverarbeiten
X_train_processed = preprocess.fit_transform(X_train)
X_test_processed = preprocess.transform(X_test)

# Modelle definieren und trainieren
models = {}
predictions = {}
probabilities = {}

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_processed, y_train)
models["Logistic Regression"] = lr

# 2. Decision Tree
dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)
dt.fit(X_train_processed, y_train)
models["Decision Tree"] = dt

# 3. Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_processed, y_train)
models["Random Forest"] = rf

# 4. GradientBoosting
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=RANDOM_STATE
)
gb.fit(X_train_processed, y_train)
models["GradientBoosting"] = gb

# 5. XGBoost mit Early Stopping
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=30
)

xgb.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)
models["XGBoost"] = xgb

# Predictions und Probabilities sammeln
for name, model in models.items():
    predictions[name] = model.predict(X_test_processed)
    probabilities[name] = model.predict_proba(X_test_processed)[:, 1]


### Evaluation
#Visualisierung der Ergebnisse
feature_names = (preprocess.named_transformers_['cat'].get_feature_names_out(cat_features).tolist() + 
                num_features)

# Tree Visualisierungen
visualize_trees(models, feature_names, max_depth_display=3)

visualize_boosting_trees(models, feature_names, n_trees=3)

# Test performance
print("\n=== Test-Set-Performance ===")
results = {}

for name in models.keys():
    y_pred = predictions[name]
    y_proba = probabilities[name]
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    
    results[name] = metrics
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:0.4f}")

# =============================================================================
# VISUALISIERUNGEN
# =============================================================================

#1. Model Performance Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['Accuracy', 'F1', 'ROC-AUC']
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

for i, metric in enumerate(metrics):
    model_names = list(results.keys())
    values = [results[name][metric] for name in model_names]
    
    bars = axes[i].bar(model_names, values, color=colors)
    axes[i].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    axes[i].set_ylabel(metric)
    axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis='x', rotation=45)
    
    # Werte auf Balken anzeigen
    for bar, value in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 2. ROC Curves
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for i, (name, model) in enumerate(models.items()):
    y_proba = probabilities[name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.plot(fpr, tpr, color=colors[i], lw=2, 
             label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 3. Precision-Recall Curves
plt.figure(figsize=(10, 8))

for i, (name, model) in enumerate(models.items()):
    y_proba = probabilities[name]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.plot(recall, precision, color=colors[i], lw=2, label=name)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. Feature Importance (für baumbasierte Modelle)
feature_names = (preprocess.named_transformers_['cat'].get_feature_names_out(cat_features).tolist() + 
                num_features)

tree_models = ["Decision Tree", "Random Forest", "GradientBoosting"]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, name in enumerate(tree_models):
    if name in models:
        model = models[name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-15:]  # Top 15 Features
            
            axes[i].barh(range(len(top_indices)), importances[top_indices])
            axes[i].set_yticks(range(len(top_indices)))
            axes[i].set_yticklabels([feature_names[idx] for idx in top_indices])
            axes[i].set_title(f'{name} - Top 15 Features', fontweight='bold')
            axes[i].set_xlabel('Feature Importance')

plt.tight_layout()
plt.show()

# 7. Learning Curve für XGBoost (wenn verfügbar)
if hasattr(xgb, 'evals_result_'):
    results_xgb = xgb.evals_result_
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_xgb['validation_0']['auc'], label='Validation AUC')
    plt.axvline(x=xgb.best_iteration, color='red', linestyle='--', 
                label=f'Best Iteration ({xgb.best_iteration})')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.title('XGBoost Learning Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Cross-Validation Ergebnisse
print("\n=== 5-Fold-CV (AUC) für Pipeline-Modelle ===")
pipeline_models = {
    "Logistic Regression": Pipeline([("pre", preprocess), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "Decision Tree": Pipeline([("pre", preprocess), ("clf", DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=RANDOM_STATE))]),
    "Random Forest": Pipeline([("pre", preprocess), ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=20, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1))]),
    "GradientBoosting": Pipeline([("pre", preprocess), ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE))]),
    "XGBoost (ohne ES)": Pipeline([("pre", preprocess), ("clf", XGBClassifier(objective="binary:logistic", learning_rate=0.05, n_estimators=500, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1))])
}

cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, pipe in pipeline_models.items():
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = auc_scores
    print(f"{name:20s}: {auc_scores.mean():0.4f} ± {auc_scores.std():0.4f}")