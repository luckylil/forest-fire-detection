from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("rtdetr-l.pt")

result = model.predict(source="C:/Users/17923/Desktop/GTR-YOLO/dataset/valid/images",show=True,save=True)

print(result)

im2 = cv2.imread("C:/Users/17923/Desktop/ultralytics/firedata/test/images/1_mountain_lake_daytime_0140_jpg.rf.ba89f8ee8d3ce2c876553cc138196032.jpg")
results = model.predict(source=im2, show=True, save=True)
print(results)