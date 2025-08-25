# Εισαγωγή βιβλιοθηκών
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
warnings.filterwarnings("ignore")

#  Επιλογές εκτέλεσης
RUN_ONLY = "All"   # Επιλογές: "All", "RandomForest", "XGBoost", "TabNet"
SAFE_WRITE = True     # Αν True, δεν αντικαθιστά υπάρχοντα αρχεία – προσθέτει _NEW

#  Φόρτωμα dataset
df = pd.read_csv("D:/features/features_dataset_multiclass.csv")

#  Συνδυασμοί labels για classification
LABEL_COMBOS = [
    ("sms2", "banking"),
    ("sms2", "benign"),
    ("banking", "benign"),
    ("sms2", "banking", "benign")
]

#  Ορισμός ταξινομητών
CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "TabNet": TabNetClassifier(
        n_d=64, n_a=64, n_steps=7,
        gamma=1.3, lambda_sparse=0.0001,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.03),
        verbose=0,
        seed=42
    )
}

#  Δημιουργία φακέλων αποθήκευσης
base_path = "D:/classification_results"
os.makedirs(f"{base_path}/models", exist_ok=True)
os.makedirs(f"{base_path}/reports", exist_ok=True)
os.makedirs(f"{base_path}/plots", exist_ok=True)

#  Συνάρτηση για Confusion Matrix
def plot_conf_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Συνάρτηση που δημιουργεί γράφημα ράβδων για τη σημαντικότητα χαρακτηριστικών
def plot_feature_importance(importances, features, model_name, label_combo):
    plt.figure(figsize=(8, 4))
    pd.Series(importances, index=features).sort_values(ascending=False).plot(kind="bar")
    plt.title(f"Feature Importance – {model_name} ({label_combo})")
    plt.ylabel("Importance")
    plt.tight_layout()
    save_path = f"{base_path}/plots/{model_name}_{label_combo}_importance.png"
    plt.savefig(save_path)
    plt.close()


#  Εκπαίδευση για κάθε συνδυασμό labels
for labels in LABEL_COMBOS:
    label_name = "_VS_".join(labels)
    subset = df[df["label"].isin(labels)].copy()

    # X = χαρακτηριστικά εισόδου, y = ετικέτες (στόχοι)
    y = subset["label"]
    X = subset.drop(columns=["label"])

    #  Train-test split
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    #  Κωδικοποίηση μετά το split, για συγχρονισμένους δείκτες
    y_train_tabnet, class_names = pd.factorize(y_train_raw)
    y_test_tabnet = pd.Series([class_names.tolist().index(lbl) for lbl in y_test_raw])

    for name, model in CLASSIFIERS.items():
        if RUN_ONLY != "All" and name != RUN_ONLY:
            continue

        print(f"\n Εκπαίδευση {name} για: {label_name}")

        if name == "TabNet":
            #  TabNet χρειάζεται αριθμητικά labels (NumPy arrays)
            model.fit(
                X_train.values, y_train_tabnet,
                eval_set=[(X_test.values, y_test_tabnet)],
                patience=10
            )

            # Υπολογισμός και αποθήκευση της σχετικής σημαντικότητας των χαρακτηριστικών (feature importance)
            plot_feature_importance(model.feature_importances_, X_train.columns, name, label_name)

            #  Μετατροπή από index σε πραγματικά string labels
            preds = class_names[model.predict(X_test.values).ravel()]

        elif name == "XGBoost":
            #  XGBoost δεν δέχεται string labels οπότε κάνουμε μετατροπή
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_raw) # π.χ. 'sms2' -> 0, 'banking'-> 1
            y_test_encoded = le.transform(y_test_raw)

            #  Εκπαίδευση XGBoost πάνω στα numeric labels
            model.fit(X_train, y_train_encoded)

            # Υπολογισμός και αποθήκευση της σχετικής σημαντικότητας των χαρακτηριστικών (feature importance)
            plot_feature_importance(model.feature_importances_, X_train.columns, name, label_name)

            #  Προβλέψεις με επιστροφή σε string labels
            preds = le.inverse_transform(model.predict(X_test))

        else:
            #  Για Random Forest δεν χρειάζεται τίποτα, δουλεύει με string labels
            model.fit(X_train, y_train_raw)

            # Υπολογισμός και αποθήκευση της σχετικής σημαντικότητας των χαρακτηριστικών (feature importance)
            plot_feature_importance(model.feature_importances_, X_train.columns, name, label_name)

            preds = model.predict(X_test)

        #  Υπολογισμός ακρίβειας και αναφοράς
        acc = accuracy_score(y_test_raw, preds)
        report = classification_report(y_test_raw, preds, zero_division=0)

        #  Αποθήκευση Confusion Matrix plot
        plot_path = f"{base_path}/plots/{name}_{label_name}_conf_matrix.png"
        if SAFE_WRITE and os.path.exists(plot_path):
            base_plot = plot_path
            if base_plot.lower().endswith(".png"):
                base_plot = base_plot[:-4]
            suffix = "_NEW"
            counter = 1
            while os.path.exists(f"{base_plot}{suffix}.png"):
                suffix = f"_NEW{counter}"
                counter += 1
            plot_path = f"{base_plot}{suffix}.png"
        plot_conf_matrix(y_test_raw, preds, class_names, plot_path)

        #  Αποθήκευση μοντέλου (μόνο αν είναι TabNet)
        if name == "TabNet":
            #  Αποθήκευση TabNet σε .zip (δεν υποστηρίζει joblib)
            model_basename = f"{base_path}/models/{name}_{label_name}"

            if SAFE_WRITE and os.path.exists(model_basename + ".zip"):
                base = model_basename
                suffix = "_NEW"
                counter = 1
                while os.path.exists(f"{base}{suffix}.zip"):
                    suffix = f"_NEW{counter}"
                    counter += 1
                model_basename = f"{base}{suffix}"

            # Το TabNet θα προσθέσει αυτόματα την κατάληξη .zip
            model.save_model(model_basename)

        else:
            model_path = f"{base_path}/models/{name}_{label_name}.pkl"
            if SAFE_WRITE and os.path.exists(model_path):
                base_model = model_path
                if base_model.lower().endswith(".pkl"):
                    base_model = base_model[:-4]
                suffix = "_NEW"
                counter = 1
                while os.path.exists(f"{base_model}{suffix}.pkl"):
                    suffix = f"_NEW{counter}"
                    counter += 1
                model_path = f"{base_model}{suffix}.pkl"
            joblib.dump(model, model_path)

        #  Αποθήκευση αναφοράς σε αρχείο
        report_path = f"{base_path}/reports/{name}_{label_name}_report.txt"
        if SAFE_WRITE and os.path.exists(report_path):
            base_report = report_path
            if base_report.lower().endswith(".txt"):
                base_report = base_report[:-4]
            suffix = "_NEW"
            counter = 1
            while os.path.exists(f"{base_report}{suffix}.txt"):
                suffix = f"_NEW{counter}"
                counter += 1
            report_path = f"{base_report}{suffix}.txt"
        with open(report_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(report)

        print(f" Ολοκληρώθηκε: {name} ({label_name}) με ακρίβεια {acc:.4f}")
