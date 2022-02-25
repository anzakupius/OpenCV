from sys import implementation
from turtle import color, width
import numpy as np
import cv2
import pickle

# haar Cascade Classifiers
face_cascades = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

#Declare face recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()

# implementation of a recognizer (the train data)
recognizer.read("trainner.yml")

#Load Names from pickle (Adding names to label)
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)                       #original labels og_labels)
    labels = {v:k for k,v in og_labels.items()}         # inverting the labels


# Using the face classifier
cap = cv2.VideoCapture(0)

while (True):
 
    # Capture frame by frame
    ret, frame = cap.read()

    # Now we use the Harr cascade to detect the faces in this frame
    # Before detecting the faces  in the frames, we have to convert it to grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # We use faces variable to find faces in the frame whether video or pictures
    faces = face_cascades.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

    # To iterate through this faces
    for (x, y, w, h) in faces:
        #print(x,y,w,h) 
        # Region Of Interest for gray is equal to the location of the frame "[y:y+h, x:x+w]"
        roi_gray = gray[y:y+h, x:x+w]    # (ycord_start, ycord_end)

        #Region of interest for the color
        roi_color = frame[y:y+h, x:x+w]     # (xcord_start, xcord_end)

        #TO recognise Region of Interest. 
        # Reognizer? Deep learned Model predict Keras, tensorflow, pytorch scikit learn

        #TO run a prediction on the roi
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])      #print labels
            font = cv2.FONT_HERSHEY_SIMPLEX     #Font type
            name = labels[id_]
            color = (255, 255, 255)         # color white
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


        # To save the image
        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray) # write the capture portion, the roi_ray and img_item

        # Draw A Rectangle
        color = (255, 0, 0)         # BGR 0 - 255
        stroke = 2                  # How thick you want the line to be
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)      # To draw on the frame
         

    #Display resulting frame 
    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()