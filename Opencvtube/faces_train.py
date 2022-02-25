import cv2
import os
import numpy as np
from PIL import Image           #PIL is Python Image Library
#from sys import path_hooks
import pickle


# path to the faces_traimport cv2in file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Image directory path
image_dir = os.path.join(BASE_DIR, "images")

# haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

#Declare face recognizer 
# Incase you run into an error module 'cv2' has no attribute 'face' then do this on a terminal "pip install opencv-contrib-python" 
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

# To see the images in those files
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # iterate through the files
        if file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            
            #To give it a label and the lower() to change everything on that label to lower case
            label = os.path.basename(root).replace(" ", "-").lower()
            # It's going to print the file or file path
            #print(label, path)

            # Creating a trainning Label
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)


            #y_labels.append(label)      # some number
            #x_train.append(path)        # verify this image, and turn into a numpy array and gray

            # Training image to numpy array
            pil_image = Image.open(path).convert("L")   # .convert("L") convert the pictures to grayscale
            size = (550, 550)                           # Resizing the images
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            # convert it into numpy array (converst images to numbers)
            image_array = np.array(final_image, "uint8")
            #print(image_array)

            # Region of Interest in Training Data
            # Detector. doing the face detection inside the actual image
            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            # training data
            for (x,y,w,h) in faces:
                roi =image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#Using Pickle to save Label Ids
#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

#Train the OpenVC Recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")