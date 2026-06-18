import fiftyone as fo
import os

dataset = fo.load_dataset("weight")
print("Dataset size:", len(dataset))
session = fo.launch_app(dataset)
session.wait()