#!/usr/bin/env python
# coding: utf-8
'''
Bank-Marketing – Robuste Benchmarks mit Hyperparameter-Optimierung
==================================================================
Verwendete Modelle
------------------
LogisticRegression · DecisionTree · RandomForest · GradientBoosting · XGBoost

Methodik
--------
* Pipeline (Vorverarbeitung ohne Leakage)
* RandomizedSearchCV (30 Suchpunkte, 5-fach Stratified CV, ROC-AUC als Score)
* class_weight / scale_pos_weight gegen Klassenschiefe
* Reproduzierbarkeit über RANDOM_STATE
'''

import os, zipfile, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# --------------------------------------------------------------------------
# 1) Daten laden
# --------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_URL  = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
CSV_PATH  = 'bank-full.csv'
if not Path(CSV_PATH).exists():
    urllib.request.urlretrieve(DATA_URL, 'bank.zip')
    with zipfile.ZipFile('bank.zip') as zf:
        zf.extract(CSV_PATH)
    os.remove('bank.zip')

df = pd.read_csv(CSV_PATH, sep=';')
X = df.drop('y', axis=1)
y = df['y'].map({'yes':1, 'no':0}).astype(np.int8)

# Imbalance-Kennzahl für XGBoost
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos
print(f'Class ratio (neg/pos): {neg}/{pos}  ->  scale_pos_weight ≈ {scale_pos_weight:.1f}')

cat_cols  = X.select_dtypes(include='object').columns.tolist()
num_cols  = X.select_dtypes(exclude='object').columns.tolist()

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(),                       num_cols)
])

# --------------------------------------------------------------------------
# 2) Train/Test-Split
# --------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# --------------------------------------------------------------------------
# 3) Modell-Definition & Suchräume
# --------------------------------------------------------------------------
N_ITER, CV_FOLDS = 30, 5
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

search_space = {
    'LogReg': {
        'estimator': LogisticRegression(max_iter=5000, n_jobs=-1, random_state=RANDOM_STATE),
        'params': {
            'estimator__C'            : np.logspace(-3, 3, 20),
            'estimator__class_weight' : [None, 'balanced'],
            'estimator__solver'       : ['lbfgs', 'liblinear'],
        }
    },
    'DecisionTree': {
        'estimator': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {
            'estimator__max_depth'        : [None, 5, 10, 20, 30],
            'estimator__min_samples_split': [2, 5, 10, 20],
            'estimator__min_samples_leaf' : [1, 2, 5, 10],
            'estimator__criterion'        : ['gini', 'entropy'],
            'estimator__class_weight'     : [None, 'balanced'],
        }
    },
    'RandomForest': {
        'estimator': RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        'params': {
            'estimator__n_estimators'     : [200, 400, 600, 800],
            'estimator__max_depth'        : [None, 10, 20, 30],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf' : [1, 2, 5],
            'estimator__max_features'     : ['sqrt', 'log2', 0.7],
            'estimator__class_weight'     : [None, 'balanced'],
        }
    },
    'GradientBoosting': {
        'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'estimator__n_estimators' : [200, 400, 600, 800],
            'estimator__learning_rate': np.linspace(0.01, 0.2, 20),
            'estimator__max_depth'    : [2, 3, 4, 5],
            'estimator__subsample'    : [0.6, 0.8, 1.0],
        }
    },
    'XGBoost': {
        'estimator': XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        'params': {
            'estimator__n_estimators'     : [300, 600, 1000, 1500],
            'estimator__learning_rate'    : np.linspace(0.01, 0.2, 20),
            'estimator__max_depth'        : [3, 4, 5, 6],
            'estimator__subsample'        : [0.6, 0.8, 1.0],
            'estimator__colsample_bytree' : [0.6, 0.8, 1.0],
            'estimator__reg_lambda'       : [1.0, 3.0, 5.0, 10.0],
            'estimator__scale_pos_weight': [scale_pos_weight],
        }
    },
}

# --------------------------------------------------------------------------
# 4) Hyperparameter-Suche + Training
# --------------------------------------------------------------------------
best_models, summary = {}, []

for name, cfg in search_space.items():
    print(f'\n🟢  Suche für {name} …')
    pipe = Pipeline([('prep', preprocess), ('estimator', cfg['estimator'])])

    search = RandomizedSearchCV(
        pipe,
        cfg['params'],
        n_iter=N_ITER,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    search.fit(X_tr, y_tr)
    best_models[name] = search.best_estimator_

    print(f'Best ROC-AUC (CV): {search.best_score_:.4f}')
    print('Beste Parameter :', search.best_params_)

# --------------------------------------------------------------------------
# 5) Test-Set-Performance
# --------------------------------------------------------------------------
for name, model in best_models.items():
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    summary.append({
        'Modell'  : name,
        'Accuracy': accuracy_score(y_te, y_pred),
        'F1'      : f1_score(y_te, y_pred),
        'ROC-AUC' : roc_auc_score(y_te, y_proba),
    })

results = pd.DataFrame(summary).sort_values('ROC-AUC', ascending=False)
print('\n===== Test-Set-Ergebnisse =====')
print(results.to_string(index=False, float_format='%.4f'))

# --------------------------------------------------------------------------
# 6) ROC-Kurven
# --------------------------------------------------------------------------
plt.figure(figsize=(8,6))
for name, model in best_models.items():
    RocCurveDisplay.from_estimator(model, X_te, y_te, name=name)
plt.title('ROC-Kurven (Test-Set)')
plt.show()

# --------------------------------------------------------------------------
# 7) Modelle serialisieren
# --------------------------------------------------------------------------
from joblib import dump
from pathlib import Path
from datetime import datetime

stamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
models_dir = Path('saved_models')
models_dir.mkdir(exist_ok=True)

for name, model in best_models.items():
    fname = models_dir / f'{name.replace(" ", "_")}_{stamp}.joblib'
    dump(model, fname)            #  ➜  vollständige Pipeline!
    print(f'💾  {name} gespeichert unter  {fname}')