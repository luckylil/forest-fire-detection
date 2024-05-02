from ultralytics import YOLO
import torch.optim as optim
# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8-improve.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")

from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
#model = RTDETR('rtdetr-l.pt')

# Display model information (optional)
#model.info()

# Use the model
result=model.train(data='coco8.yaml', epochs=100, imgsz=640, batch=16, workers=0, device=0,optimizer='SGD',lr0=0.00261)  # train the mode


result = model.val()  # evaluate model performance on the validation set


