from pathlib import Path
import zipfile, urllib.request, glob, os, re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

# -------------------------------
# 0) globale Konstanten
# -------------------------------
RANDOM_STATE = 42
MODELS_DIR = Path("saved_models")
CSV_PATH     = "bank-full.csv"

# -------------------------------
# 1) Modelle laden
# -------------------------------
model_files = sorted(glob.glob(str(MODELS_DIR / "*.joblib")))
if not model_files:
    raise FileNotFoundError(
        f"Keine .joblib-Dateien in {MODELS_DIR}. "
        "Bitte zuerst trainieren und speichern!"
    )

models = {Path(f).stem: load(f) for f in model_files}
print(f"✅ {len(models)} Modelle geladen.")

# Hilfsfunktion für hübschere Bezeichnungen
_underscore_timestamp = re.compile(r"_[0-9]{8}_[0-9]{6}$")

def pretty_label(model_name: str) -> str:
    """Entfernt Zeitstempel & Unterstriche → besser lesbare Achsenbeschriftung."""
    return _underscore_timestamp.sub("", model_name).replace("_", " ")

display_names = {name: pretty_label(name) for name in models}

# -------------------------------
# 2) Datensatz laden (UCI Bank Marketing)
# -------------------------------
if not Path(CSV_PATH).exists():
    URL = (
        "4"
    )
    print("⇩ Lade Datensatz …")
    urllib.request.urlretrieve(URL, "bank.zip")
    with zipfile.ZipFile("bank.zip") as zf:
        zf.extract(CSV_PATH)
    os.remove("bank.zip")

print("📄 Lese CSV …")
df = pd.read_csv(CSV_PATH, sep=";")
X = df.drop("y", axis=1)
y = df["y"].map({"yes": 1, "no": 0}).astype(np.int8)

# -------------------------------
# 3) Reproduzierbarer Train/Test-Split
# -------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -------------------------------
# 4) Benchmarks berechnen
# -------------------------------
results_rows = []            # für DataFrame
probas: dict[str, np.ndarray] = {}  # für ROC/PR‑Kurven

for internal_name, model in models.items():
    label = display_names[internal_name]
    print(f"⏳ Bewerte {label} …")

    # Vorhersagen & Wahrscheinlichkeiten
    y_pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_te)
        y_proba = (dec - dec.min()) / (dec.max() - dec.min())
    else:
        y_proba = y_pred.astype(float)

    probas[label] = y_proba

    results_rows.append(
        {
            "Modell": label,
            "Accuracy": accuracy_score(y_te, y_pred),
            "F1": f1_score(y_te, y_pred),
            "ROC-AUC": roc_auc_score(y_te, y_proba),
        }
    )

scores = (
    pd.DataFrame(results_rows)
    .sort_values("ROC-AUC", ascending=False)
    .reset_index(drop=True)
    .round(4)
)

print("\n===== Test-Set-Ergebnisse =====")
print(scores.to_string(index=False))

# =============================================================================
# 5) VISUALISIERUNGEN
# =============================================================================

plt.style.use("ggplot")  # angenehme Grundoptik
palette_bars = plt.cm.Set2(np.linspace(0, 1, len(scores)))  # Pastellfarben

# 5.1 Metrik-Vergleich (Balkendiagramme)
metrics = ["Accuracy", "F1", "ROC-AUC"]
for metric in metrics:
    plt.figure(figsize=(8, 4))
    plt.bar(scores["Modell"], scores[metric], color=palette_bars)
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.title(f"{metric} – Vergleich der Modelle")
    plt.xticks(rotation=45, ha="right")
    for x, v in zip(scores["Modell"], scores[metric]):
        plt.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()

# 5.2 ROC‑Kurven
colors_roc = plt.cm.tab10(np.linspace(0, 1, len(probas)))
plt.figure(figsize=(7, 6))
for (label, y_proba), col in zip(probas.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    auc = roc_auc_score(y_te, y_proba)
    plt.plot(fpr, tpr, lw=1.8, label=f"{label} (AUC = {auc:.3f})", color=col)
plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC‑Kurven (alle Modelle)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5.3 Precision‑Recall‑Kurven
plt.figure(figsize=(7, 6))
for (label, y_proba), col in zip(probas.items(), colors_roc):
    precision, recall, _ = precision_recall_curve(y_te, y_proba)
    plt.plot(recall, precision, lw=1.8, label=label, color=col)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision‑Recall‑Kurven")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5.4 Feature Importance (optional) ---------------------------
print("\n===== Feature Importances =====")
for internal_name, model in models.items():
    label = display_names[internal_name]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) != X.shape[1]:
            print(f"(⚠ {label}: Feature Importances nicht darstellbar – andere Dim.)")
            continue

        idx_sorted = np.argsort(importances)[-15:][::-1]
        plt.figure(figsize=(6, 4))
        plt.barh(
            np.arange(len(idx_sorted)),
            importances[idx_sorted][::-1],
            color=plt.cm.viridis(np.linspace(0, 1, len(idx_sorted)))[::-1],
        )
        plt.yticks(np.arange(len(idx_sorted)), X.columns[idx_sorted][::-1], fontsize=8)
        plt.xlabel("Importance")
        plt.title(f"{label} – Top 15 Features")
        plt.tight_layout()
        plt.show()
    else:
        print(f"(ℹ {label}: keine feature_importances_)" )
