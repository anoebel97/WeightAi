import fiftyone as fo
import os
import math

def weight_to_bucket(w, base=40, ratio=1.25):
    if w is None:
        return "0"
    w = max(float(w), base)
    n = math.floor(math.log(w / base, ratio))
    lo = base * (ratio ** n)
    hi = base * (ratio ** (n + 1))
    return f"{int(round(lo))}-{int(round(hi))}"

dataset = fo.load_dataset("weight")
new_images = "new_images"
new_samples = [
    fo.Sample(filepath=os.path.join(new_images, f))
    for f in os.listdir(new_images)
    if f.endswith((".jpg", ".png", ".jpeg"))
]
dataset.add_samples(new_samples)
for s in dataset:
    s["weight_bucket"] = weight_to_bucket(s["weight"])
    s.save()
session = fo.launch_app(dataset)
session.wait()