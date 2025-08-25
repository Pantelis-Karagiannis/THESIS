#  Εισαγωγή απαραίτητων βιβλιοθηκών
from pymongo import MongoClient
import pandas as pd
import os

#  Συνάρτηση για ασφαλή μετατροπή σε ακέραιο αριθμό
def safe_int(value, default=-1):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

#  Σύνδεση με τη MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["mobsf_analysis"]
collection = db["static_analysis"]

#  Συνάρτηση για εξαγωγή χαρακτηριστικών από κάθε έγγραφο JSON
def extract_features(doc):
    permissions = doc.get("permissions", {})
    total_permissions = len(permissions)

    # Υπολογίζει τον αριθμό των dangerous permissions
    dangerous_permissions = sum(1 for p in permissions.values() if p.get("status") == "dangerous")

    # Υπολογίζει το ποσοστό των dangerous permissions
    dangerous_ratio = dangerous_permissions / total_permissions if total_permissions else 0

    # Δημιουργία λεξικού με τα 20 χαρακτηριστικά
    features = {
        # Ασφαλής μετατροπή των SDK σε αριθμό
        "min_sdk": safe_int(doc.get("min_sdk", -1)),
        "target_sdk": safe_int(doc.get("target_sdk", -1)),

        "num_permissions": total_permissions,
        "dangerous_ratio": dangerous_ratio,

        # Έλεγχος για παρουσία συγκεκριμένων permissions
        "SEND_SMS": int("android.permission.SEND_SMS" in permissions),
        "READ_SMS": int("android.permission.READ_SMS" in permissions),
        "WRITE_SMS": int("android.permission.WRITE_SMS" in permissions),
        "starts_on_boot": int("android.permission.RECEIVE_BOOT_COMPLETED" in permissions),

        # Έλεγχος για receivers που περιέχουν "sms"
        "has_sms_receiver": int(any("sms" in r.lower() for r in doc.get("receivers", []))),

        # Μετρικά από το manifest και τον κώδικα
        "manifest_high": int(doc.get("manifest_analysis", {}).get("manifest_summary", {}).get("high", 0)),
        "manifest_warning": int(doc.get("manifest_analysis", {}).get("manifest_summary", {}).get("warning", 0)),
        "code_high": int(doc.get("code_analysis", {}).get("summary", {}).get("high", 0)),
        "code_warning": int(doc.get("code_analysis", {}).get("summary", {}).get("warning", 0)),

        # Πλήθος στοιχείων
        "num_receivers": len(doc.get("receivers", [])),
        "num_services": len(doc.get("services", [])),
        "num_libraries": len(doc.get("libraries", [])),
        "num_trackers": int(doc.get("trackers", {}).get("detected_trackers", 0)),
        "num_domains": len(doc.get("domains", [])),
        "num_urls": sum(len(d.get("urls", [])) for d in doc.get("domains", [])),

        # Έλεγχος για παρουσία private IP
        "has_private_ip": int(any(
            "192.168." in u or "10." in u or "127.0.0.1" in u
            for d in doc.get("domains", []) for u in d.get("urls", [])
        )),

        # Ετικέτα (benign / banking / sms2)
        "label": doc.get("label", "unknown")
    }

    return features

#  Εξαγωγή μόνο των εγγραφών που ανήκουν σε μια από τις κατηγορίες
query = {"label": {"$in": ["benign", "banking", "sms2"]}}
data = [extract_features(doc) for doc in collection.find(query)]

#  Δημιουργία pandas DataFrame από τα δεδομένα
df = pd.DataFrame(data)

#  Δημιουργία φακέλου "features" αν δεν υπάρχει ήδη
os.makedirs("D:/features", exist_ok=True)

#  Αποθήκευση του DataFrame σε CSV αρχείο
df.to_csv("D:/features/features_dataset_multiclass.csv", index=False)

#  Αποθήκευση του DataFrame και σε pickle αρχείο
df.to_pickle("D:/features/features_dataset_multiclass.pkl")

#  Επιβεβαίωση ολοκλήρωσης
print(" Έγινε εξαγωγή των χαρακτηριστικών για multiclass και αποθήκευση στα αρχεία CSV και Pickle")
