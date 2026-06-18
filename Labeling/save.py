import fiftyone as fo
import json
import math

def weight_to_bucket(w, base=40, ratio=1.25):
    if w is None:
        return 0
    w = max(float(w), base)
    n = math.floor(math.log(w / base, ratio))
    lo = base * (ratio ** n)
    hi = base * (ratio ** (n + 1))
    return f"{int(round(lo))}-{int(round(hi))}"

ds = fo.load_dataset("weight")

export = []

for s in ds:
    weight = 0
    export.append({
        "filepath": s.filepath,
        "pose": s["pose"],
        "viewport": s["viewport"],
        "weight": s["weight"],
        "height": s["height"],
        "subject": s["subject"],
        "url": s["url"],
        "weight_bucket": s["weight_bucket"],
    })

with open("export.json", "w", encoding="utf-8") as f:
    json.dump(export, f, indent=2)
