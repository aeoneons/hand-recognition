import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe.framework.formats import landmark_pb2
from threading import Thread
from queue import Queue
import pyautogui
import os
import math
from flask import Flask




modelPath = "C:\\Users\\Jacks\\OneDrive\\Documents\\GitHub\\hand-recognition\\hand_landmarker.task"


mpDrawing = mp.solutions.drawing_utils
mpStyle = mp.solutions.drawing_styles
mpHands = mp.solutions.hands
baseOption = mp.tasks.BaseOptions
handLandMarker = mp.tasks.vision.HandLandmarker
handLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
handLandMarkerResult = mp.tasks.vision.HandLandmarkerResult
visionRunningMode = mp.tasks.vision.RunningMode
hand = []





    
options = handLandMarkerOptions(base_options = baseOption(model_asset_path = modelPath),
                                running_mode = visionRunningMode.IMAGE,
                                num_hands = 1)
            




        
landmarker = handLandMarker.create_from_options(options) 


def proccesFramePhoto(frame):
    global hand
    

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #coverts the frame to colored

    mpImage = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgbFrame) #converts frame to mediapipe format

    result = landmarker.detect(mpImage) #detects the hand async with the mp image, gives a list of the normalized x,y,z values of landmarks

    if result and result.hand_landmarks: #checks if there is a iamge and if there is a hand in that image
        
        hand = result.hand_landmarks[0]
        for hand in result.hand_landmarks: #latestResult.hand_landmarks is in list format with each handland cordinate its own list. iterates throug that list
            
            #landmark drawing
            landmarkList = landmark_pb2.NormalizedLandmarkList( #nomlizes each cordinates and puts in the object that mpDrawing.draw_landmarks wants
                landmark=[landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand]) #list comeprenshion that makes the list of x,y,z to mediapipe format
            
            mpDrawing.draw_landmarks( 
                rgbFrame,
                landmarkList,
                mpHands.HAND_CONNECTIONS
            )
    drawnImg = cv2.cvtColor(rgbFrame, cv2.COLOR_RGB2BGR)

           
    return (drawnImg, [(landmark.x, landmark.y, landmark.z) for landmark in hand])
        




