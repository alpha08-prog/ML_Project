import os
import requests

BASE_URL = "https://physionet.org/files/eegmat/1.0.0/"
DATA_FOLDER = "Data/eegmat"

records_path = os.path.join(DATA_FOLDER, "RECORDS")

with open(records_path) as f:
    files = f.read().splitlines()

for file_name in files:
    url = BASE_URL + file_name
    save_path = os.path.join(DATA_FOLDER, file_name)

    if os.path.exists(save_path):
        print(f"Skipping (exists): {file_name}")
        continue

    print(f"Downloading: {file_name}")

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed: {file_name} (status {r.status_code})")
    except Exception as e:
        print(f"Error: {file_name} → {e}")