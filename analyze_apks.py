#  Εισαγωγή βιβλιοθηκών
import requests
import json
import os
import time
from pymongo import MongoClient

#  MobSF API Key
api_key = "d648b459c9f85817161bf3e4e2df4dbfe0dd335dbbc814387e73a70a89a6676c"

#  Ανέβασμα APK στο MobSF
def upload_apk(file_path, filename):
    url = "http://localhost:8000/api/v1/upload"
    headers = {"X-Mobsf-Api-Key": api_key}
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, "application/vnd.android.package-archive")}
            response = requests.post(url, files=files, headers=headers)
        if response.status_code == 200:
            return response.json().get("hash")
        print("Upload failed:", response.text)
    except Exception as e:
        print(f" Σφάλμα με το αρχείο:\n{file_path}\n➡ {e}")
    return None

# Σάρωση του APK
def scan_apk(scan_id):
    url = "http://localhost:8000/api/v1/scan"
    headers = {"X-Mobsf-Api-Key": api_key}
    data = {"hash": scan_id}
    response = requests.post(url, headers=headers, data=data)
    return response.json() if response.status_code == 200 else None

#  Λήψη αναφοράς JSON
def get_scan_results(scan_id):
    url = "http://localhost:8000/api/v1/report_json"
    headers = {"X-Mobsf-Api-Key": api_key}
    data = {"hash": scan_id}
    response = requests.post(url, data=data, headers=headers)
    return response.json() if response.status_code == 200 else None

#  Αποθήκευση JSON τοπικά
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

#  Ανάλυση και αποθήκευση ενός APK
def analyze_and_store(file_path, apk_name, label):
    print(f"\n Ανάλυση: {apk_name} ({label})")
    upload_id = upload_apk(file_path, apk_name)
    if not upload_id:
        return

    scan_data = scan_apk(upload_id)
    if not scan_data:
        return

    analysis_data = get_scan_results(upload_id)
    if not analysis_data:
        return

    structured_data = {
        "hash": upload_id,
        "file_name": apk_name,
        "label": label,
        "package_name": analysis_data.get("package_name"),
        "main_activity": analysis_data.get("main_activity"),
        "target_sdk": analysis_data.get("target_sdk"),
        "min_sdk": analysis_data.get("min_sdk"),
        "permissions": analysis_data.get("permissions", []),
        "activities": analysis_data.get("activities", []),
        "services": analysis_data.get("services", []),
        "receivers": analysis_data.get("receivers", []),
        "providers": analysis_data.get("providers", []),
        "libraries": analysis_data.get("libraries", []),
        "trackers": analysis_data.get("trackers", []),
        "domains": analysis_data.get("urls", []),
        "manifest_analysis": analysis_data.get("manifest_analysis", {}),
        "code_analysis": analysis_data.get("code_analysis", {}),
        "analysis_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    output_dir = os.path.join("D:\\json_results", label)
    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.join(output_dir, upload_id + ".json")
    save_to_json(structured_data, json_filename)

    client = MongoClient("mongodb://localhost:27017/")
    collection = client["mobsf_analysis"]["static_analysis"]

    if not collection.find_one({"hash": upload_id}):
        collection.insert_one(structured_data)
        print("Αποθηκεύτηκε στη MongoDB")
    else:
        print("Το αρχείο υπάρχει ήδη — δεν αποθηκεύτηκε ξανά")

    print(f"Ολοκληρώθηκε η ανάλυση για: {upload_id}")

#  Εκτέλεση για 3 κατηγορίες (benign, banking, sms2)
SAMPLES_FOLDER = "D:\\apks"
MAX_SAMPLES = 500

benign_count = 0
banking_count = 0
sms2_count = 0

for root, _, files in os.walk(SAMPLES_FOLDER):
    for apk_file in files:
        if not apk_file.endswith(".apk"):
            continue
        apk_path = os.path.join(root, apk_file)

        if "benign" in root.lower() and benign_count < MAX_SAMPLES:
            analyze_and_store(apk_path, apk_file, "benign")
            benign_count += 1

        elif "banking" in root.lower() and banking_count < MAX_SAMPLES:
            analyze_and_store(apk_path, apk_file, "banking")
            banking_count += 1

        elif "sms2" in root.lower() and sms2_count < MAX_SAMPLES:
            analyze_and_store(apk_path, apk_file, "sms2")
            sms2_count += 1

        if benign_count >= MAX_SAMPLES and banking_count >= MAX_SAMPLES and sms2_count >= MAX_SAMPLES:
            print("\n Συλλέχθηκαν 500 benign, 500 banking και 500 sms2 APKs.")
            break
