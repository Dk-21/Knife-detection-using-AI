import numpy as np
import argparse
import os
import time
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import cvlib as cv
from cvlib.object_detection import draw_bbox

print("Abhinn")


# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help = 'images/street.jpg')
# ap.add_argument('-c', '--config', required=True,
#                 help = 'yolo-coco')
# ap.add_argument('-w', '--weights', required=True,
#                 help = 'path to yolo pre-trained weights')
# ap.add_argument('-cl', '--classes', required=True,
#                 help = 'path to text file containing class names')
# args = ap.parse_args()
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "C:/Users/Abhinn/Desktop/UTA/ML_CSE6363/yolo-object-detection/yolo-object-detection/images/Street.jpg", required=True,
# 	help="C:/Users/Abhinn/Desktop/UTA/ML_CSE6363/yolo-object-detection/yolo-object-detection/images/Street.jpg")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
# ap.add_argument("-y", "C:/Users/Abhinn/Desktop/UTA/ML_CSE6363/yolo-object-detection/yolo-object-detection/yolo-coco", required=True,
# 	help="C:/Users/Abhinn/Desktop/UTA/ML_CSE6363/yolo-object-detection/yolo-object-detection/yolo-coco")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="C:/Users/Abhinn/Desktop/UTA/ML_CSE6363/yolo-object-detection/yolo-object-detection/images/Street.jpg")
# ap.add_argument("-y", "--yolo", required=True,
# 	help="C:/Users/Abhinn/Desktop/UTA/ML_CSE6363/yolo-object-detection/yolo-object-detection/yolo-coco")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())

print("Abhinn")

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# try:
#     saver.restore(sess, os.getcwd() + '/model.ckpt')
#     print 'load from past checkpoint'
# except Exception as e:
#     print e
#     try:
#         print 'load yolo small'
#         saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
#         print 'loaded from YOLO small pretrained'
#     except Exception as e:
#         print e
#         print 'exit, atleast need a pretrained model'
#         exit(0)
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    # def load_weights(self, src_loader):
    #     val = src_loader([self.presenter])
    #     if val is None: return None
    #     else: return val.w

configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Our Image>loading
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.3f} seconds".format(end - start))


boxes = []
confidences = []
classIDs = []
# if os.name =='nt' :
#     ext_modules=[
#         Extension("darkflow.cython_utils.nms",
#             sources=["darkflow/cython_utils/nms.pyx"],
#             #libraries=["m"] # Unix-like specific
#             include_dirs=[numpy.get_include()]
#         ),        
#         Extension("darkflow.cython_utils.cy_yolo2_findboxes",
#             sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
#             #libraries=["m"] # Unix-like specific
#             include_dirs=[numpy.get_include()]
#         ),
#         Extension("darkflow.cython_utils.cy_yolo_findboxes",
#             sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
#             #libraries=["m"] # Unix-like specific
#             include_dirs=[numpy.get_include()]
#         )
#     ]

print("Checker")
# def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

#     label = str(classes[class_id])

#     color = COLORS[class_id]

#     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

#     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:

		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > args["confidence"]:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)


idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

if len(idxs) > 0:
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
# for i in indices:
#     i = i[0]
#     box = boxes[i]
#     x = box[0]
#     y = box[1]
#     w = box[2]
#     h = box[3]
#     draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
cv2.imshow("Image", image)
# cv2.imshow("image", image)
# cv2.waitKey()
print("Checking")

cv2.waitKey(0)
# cv2.imwrite("object-detection.jpg", image)
# cv2.destroyAllWindows()
print('It workssssssss')


#################
###############
####################
###################
####### IF THIS CODE IS WORKING, IT'S ORIGINAL
####### IF ITS NOT, IT'S COPIED
#           _
#      ..__(.)> (Hey, Wassup!!)
#       \____)   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
############################
####################
######################
#################
