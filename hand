import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe.framework.formats import landmark_pb2
from threading import Thread
from queue import Queue
import pyautogui
import os

cam = cv2.VideoCapture(0)
frameWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frameWidth, frameHeight))




modelPath = "C:\\Users\\Jacks\\OneDrive\\Documents\\GitHub\\hand-recognition\\hand_landmarker.task"


mpDrawing = mp.solutions.drawing_utils
mpStyle = mp.solutions.drawing_styles
mpHands = mp.solutions.hands
baseOption = mp.tasks.BaseOptions
handLandMarker = mp.tasks.vision.HandLandmarker
handLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
handLandMarkerResult = mp.tasks.vision.HandLandmarkerResult
visionRunningMode = mp.tasks.vision.RunningMode
timestamp = 0

latestResult = None
latestTimeStamp = -1
camBool = True
counter = 0

frameQueue = Queue(maxsize=200) #queue for all the frames

def writeThread(): #function to get a frame from the queue and write it
    while True:
        frame = frameQueue.get()
        if frame is None:
            break
        out.write(frame)



def draw_result(result: handLandMarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latestResult, latestTimeStamp
    if timestamp_ms > latestTimeStamp: #only keeps the newest handlandmark results
        latestResult = result
        latestTimeStamp = timestamp_ms
    
options = handLandMarkerOptions(base_options = baseOption(model_asset_path = modelPath),
                                running_mode = visionRunningMode.LIVE_STREAM,
                                result_callback = draw_result,
                                num_hands = 2)



def thumbsup(hand): #function for decting a thumbs up
    hand4 = hand[4]

    for i in range(21): 
        difference = hand4.y - hand[i].y #finds the difference between the thumb tip y and the rest of the landmark y
        if difference > -0.03 and difference != 0.0: #checks if the thumbtip y is a decent bit bigger then the rest of the rest of the y
            return False
    return True



def flip():
    global camBool
    if camBool:
        camBool = False
    else:
        camBool = True
        
with handLandMarker.create_from_options(options) as landmarker:
    outThread = Thread(target = writeThread, daemon=True) #multithreading to stop lag
    outThread.start()
    while True:
        ret, frame = cam.read() #starts of the live strean, frame is the image in numpy format and ret is either true/false
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #coverts the frame to colored

        mpImage = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgbFrame) #converts frame to mediapipe format

        if counter % 3 == 0: #throttled so async does not fall behind and start lagging
            landmarker.detect_async(mpImage, timestamp) #detects the hand async with the mp image, gives a list of the normalized x,y,z values of landmarks
        counter += 1
        timestamp += 50
    
        if latestResult and latestResult.hand_landmarks: #checks if there is a iamge and if there is a hand in that image
            for hand in latestResult.hand_landmarks: #latestResult.hand_landmarks is in list format with each handland cordinate its own list. iterates throug that list
                
                #gesture control
                if timestamp % 1000 == 0: #throttles the function call so it will only check every second
                    if thumbsup(hand):
                        os.startfile("C:\XboxGames\Minecraft Launcher\Content\Minecraft.exe")


                


                #landmark drawing
                landmarkList = landmark_pb2.NormalizedLandmarkList( #nomlizes each cordinates and puts in the object that mpDrawing.draw_landmarks wants
                    landmark=[landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand]) #list comeprenshion that makes the list of x,y,z to mediapipe format
                mpDrawing.draw_landmarks( 
                    rgbFrame,
                    landmarkList,
                    mpHands.HAND_CONNECTIONS
                )

        if camBool:
            drawnImage = cv2.cvtColor(rgbFrame, cv2.COLOR_RGB2BGR) #converts the rgbframe that was drawn on to cv2 format so it can be shown
            if not frameQueue.full():
                frameQueue.put(drawnImage)  
            cv2.imshow("thing", drawnImage)

        key = cv2.waitKey(1)
        if key == ord('h'):
            break
        if key == ord('l'):
            flip()




frameQueue.put(None)
cam.release()
out.release()
cv2.destroyAllWindows()