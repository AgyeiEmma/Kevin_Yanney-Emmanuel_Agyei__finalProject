# Import necessary libraries
!pip install face_recognition
import numpy as np
import cv2
import face_recognition
from datetime import datetime

import os
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js

# Set the path to the directory containing images
path = '/content/drive/MyDrive/new images'

# Lists to store images and corresponding names
images = []
Names = []

# Get the list of files in the specified directory
List = os.listdir(path)

# Mount Google Drive to access files (specific to Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Loop through the list of files and load images
for i in List:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    Names.append(os.path.splitext(i)[0])

# Function to encode faces in the images
def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

# Function to update attendance in a CSV file
def update_attendance(name):
    csv_file_name = '/content/drive/MyDrive/attendance.csv'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        with open(csv_file_name, mode='r') as file:
            reader = csv.reader(file)
            attendance_data = list(reader)
    except FileNotFoundError:
        attendance_data = []

    name_exists = any(row[0] == name for row in attendance_data)

    if not name_exists:
        new_row = [name, current_time]
        attendance_data.append(new_row)

        with open(csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(attendance_data)

        print(f'{name} has been added to the attendance sheet at {current_time}.')
    else:
        print(f'{name} is already in the attendance sheet.')

# Load and encode images
encodedImages = findEncodings(images)

# Capture image from webcam in a Google Colab environment
from IPython.display import Image
filename = take_photo()
print('Saved to {}'.format(filename))
display(Image(filename))

# OpenCV capture setup
capture = cv2.VideoCapture(filename)

# Face recognition loop
while True:
    success, img = capture.read()
    if img is None or img.size == 0:
        continue

    resizedImage = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    resizedImageRGB = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(resizedImageRGB)

    if not encodedImages:
        print("No encoded images to compare.")
        continue

    faceperframe = face_locations[0]
    encodeperframe = face_recognition.face_encodings(resizedImageRGB, [faceperframe])[0]

    best_match_index = None
    min_distance = float('inf')

    for i, encodeface in enumerate(encodedImages):
        distance = face_recognition.face_distance([encodeface], encodeperframe)
        print(f"Distance to face {i}: {distance}")

        if distance < min_distance:
            min_distance = distance
            best_match_index = i

    print("Best match index:", best_match_index)
    threshold = 0.6

    if min_distance < threshold:
        name = Names[best_match_index].upper()
        print("Running this")
        print("Best match:", name)
        update_attendance(name)

    cv2_imshow(img)

    if (cv2.waitKey(1) or 0xFF == ord('q')):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
