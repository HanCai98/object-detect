from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))

# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath("yolo-tiny.h5")

detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images/5.jpg"), output_image_path=os.path.join(execution_path , "output_images/imagenew5.jpeg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], eachObject["box_points"])