# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector.model",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
		
	args = vars(ap.parse_args())

	'''
	OpenCV: huge open source library for computer vision, machine learning and image processing.
	Mainly used for processing image , videos stream to identiy object, faces, handwritings and 
	so on.

	Matplotlib is a cross-platform, data visualization and graphical plotting library
	for python. 
	
	Numpy is more like the mathematical powerhouse to perform scientic
	calculation on arrays. Together created for an alternative of Matlab with python
	power.
	'''
	# load our serialized face detector model 
	'''
	Caffee is a laser speed deep learning framework created from a prototxt files that defines
	the model architecture (i. the layer themselves.) Caffeemoddel files are binary protocol
	buffer files that can be integrated with your application for image classification and 
	image segmenation model. One of the pretrained model is res10_300x300_ssd_iter_140000.caffemodel.
	'''
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])

	'''
	Open cv dnn modeule is one the world best computer vision libraries that supports deep
	learning architecture on images and videos. It supports different models like googlenet,
	alexnet and so on, but also supports many popular deep learning frameworks like Caffee, 
	tensorflow, pytorch and so on. 
	'''
	net = cv2.dnn.readNet(prototxtPath, weightsPath) # params deep learning architecture and 
	#binary files of pretrained weights.

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])

	# load the input image from disk, clone it, and grab the image spatial
	# dimensions
	'''
	imread is a function the reads the pixel of the image in BGR format because that 
	was the most famous format at the time where cv2 project started at Intel (2002)
	and got stuck with it.
	'''
	image = cv2.imread(args["image"])
	(h, w) = image.shape[:2]

	# construct a blob from the image
	'''
	Blob: It can be considered any image whose group of pixelated values distinguishes itslef
	from the background. Here blob from image is optionally resizing and croping the image
	from enter, sutracting the mean values, scaling the blob and swapping the blue and red 
	channels.
	https://towardsdatascience.com/image-processing-blob-detection-204dc6428dd
	'''
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward() #holds an array of all the predictions  (1, 1000, 1, 1) 

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]: #i.e an anchor box contains an object
			# compute the (x, y)-coordinates of the bounding box for
			# the object 
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = image[startY:endY, startX:endX]
			face = cv2.resize(face, (224, 224))
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
	
if __name__ == "__main__":
	mask_image()
