import cv2
from ultralytics import YOLO
import json

with open("data.json", "r") as f:
    data = json.load(f)

images = [item["filepath"] for item in data]

missingBoxes = []
for image in images:
    model = YOLO("yolov8n.pt")

    results = model(image, save=True, classes=[0])
    img = cv2.imread(image)
    if len(results[0].boxes) == 0:
        missingBoxes.append(image)
        cv2.imwrite(image.replace("fiftyone", "images"), img=img)
    else:
        x1, y1, x2, y2 = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        print(x1, y1, x2, y2)
        pad = 0.10
        w = x2 - x1
        h = y2 - y1

        x1p = int(max(0, x1 - w * pad))
        y1p = int(max(0, y1 - h * pad))
        x2p = int(min(img.shape[1], x2 + w * pad))
        y2p = int(min(img.shape[0], y2 + h * pad))

        crop = img[y1p:y2p, x1p:x2p]
        cv2.imwrite(image.replace("fiftyone", "images"), crop)

print(missingBoxes)
print(len(missingBoxes))
print(len(images))

"""
box = np.array(boxes)

sam = sam_model_registry["vit_b"](checkpoint="segment_anything/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
image = cv2.imread(img_path)
predictor.set_image(image)
masks, _, _ = predictor.predict(box=box, multimask_output=False)

mask = masks[0]

overlay = image.copy()
overlay[mask] = (0, 255, 0)  # grün eingefärbt

cv2.imwrite("test.png", overlay)
"""


