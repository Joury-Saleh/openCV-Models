import cv2
import os
from pathlib import Path

img = cv2.imread('images/people.jpg')

classNames = []
classFile = 'files/thing.names'

with open(classFile , 'rt') as f: 
    classNames = f.read().rstrip('\n').split('\n')

modelWeights = 'files/frozen_inference_graph.pb'  
modelConfig = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'


net = cv2.dnn.DetectionModel(modelWeights, modelConfig)
net.setInputSize(320,230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds , confs , bbox = net.detect(img , confThreshold = 0.5)

for classID , confidence , box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img ,box,color=(0,255,0),thickness=3)
    cv2.putText(img,classNames[classID-1],
                (box[0]+15,box[1]+25),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,180),thickness=2)              

'''change the photo path from here'''
cv2.imshow("obj-detect",img)

key = cv2.waitKey(0) 
if key == 27:
    cv2.destroyAllWindows()
elif key == ord('s'):
    downloads_path = str(Path.home() / "Downloads")
    save_path = os.path.join(downloads_path, "detected_output.png")
    cv2.imwrite(save_path, img)  


