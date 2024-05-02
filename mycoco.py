
from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train85/weights/best.pt')  # load an official model
# Validate the model
metrics = model.val(data='C:/Users/17923/Desktop/GTR-YOLO/datasets/datasets/alldata.yaml', save_json=True)