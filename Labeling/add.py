import fiftyone as fo
import os
import json
import math

# ---------------------------------------------------------
# 1) Neues Dataset erzeugen
# ---------------------------------------------------------

DATASET_NAME = "weight"
IMAGE_DIR = "fiftyone"          # Ordner, in dem deine JPGs liegen
JSON_PATH = "view.json"       # Deine JSON-Struktur

ds = fo.Dataset(DATASET_NAME, persistent=True)

ds.add_sample_field("pose", fo.StringField)
ds.add_sample_field("viewport", fo.StringField)
ds.add_sample_field("weight", fo.IntField)
ds.add_sample_field("height", fo.IntField)
ds.add_sample_field("subject", fo.StringField)
ds.add_sample_field("url", fo.StringField)
ds.add_sample_field("weight_bucket", fo.StringField)

# ---------------------------------------------------------
# 3) Bucket-Funktion (nichtlinear, mathematisch)
# ---------------------------------------------------------

def weight_to_bucket(w, base=40, ratio=1.25):
    w = max(float(w), base)
    n = math.floor(math.log(w / base, ratio))
    lo = base * (ratio ** n)
    hi = base * (ratio ** (n + 1))
    return f"{int(round(lo))}-{int(round(hi))}"

# ---------------------------------------------------------
# 4) JSON laden
# ---------------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------------------------------------
# 5) Samples erzeugen
# ---------------------------------------------------------

samples = []

for key, entry in data.items():
    if "http" in entry["url"]:
        filename = key + ".jpg"
    else:
        filename = entry["url"]
    img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.isfile(img_path):
        print("WARNUNG: Bild fehlt:", img_path)
        continue

    sample = fo.Sample(filepath=img_path)

    # Labels setzen
    sample["pose"] = entry.get("pose", "unknown")
    sample["viewport"] = entry.get("viewport", "unknown")
    sample["weight"] = int(entry.get("weight", 0))
    sample["height"] = int(entry.get("height", 0))
    sample["subject"] = entry.get("subject", "unknown")
    sample["url"] = entry.get("url", "unknown")

    # Bucket automatisch berechnen
    sample["weight_bucket"] = weight_to_bucket(entry["weight"])

    samples.append(sample)

# ---------------------------------------------------------
# 6) Samples hinzufügen
# ---------------------------------------------------------

ds.add_samples(samples)

# ---------------------------------------------------------
# 7) UI starten
# ---------------------------------------------------------
session = fo.launch_app(ds)
session.wait()
