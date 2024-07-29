from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import imutils
import argparse
import time
import cv2
import numpy as np
import os
import logging
from imutils.video import VideoStream
from imutils.video import FPS
import pyttsx3

main = tkinter.Tk()
main.title("")
main.geometry("1300x1200")

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")
    

ASSETS_PATH = 'assets/'
MODEL_PATH = os.path.join(ASSETS_PATH, 'frozen_inference_graph.pb')
CONFIG_PATH = os.path.join(ASSETS_PATH, 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
LABELS_PATH = os.path.join(ASSETS_PATH, 'labels.txt')
SCORE_THRESHOLD = 0.8
NETWORK_INPUT_SIZE = (300, 300)
NETWORK_SCALE_FACTOR = 1
global filename
global train
global ga_acc, bat_acc, bee_acc
global classifier

CLASSES = []
with open("./assets/labels.txt","r")as f:
    CLASSES=[line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

logger = logging.getLogger('detector')
logging.basicConfig(level=logging.INFO)

# Reading coco labels
with open(LABELS_PATH, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
logger.info(f'Available labels: \n{labels}\n')
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading model from file
logger.info('Loading model from tensorflow...')
ssd_net = cv2.dnn.readNetFromTensorflow(model=MODEL_PATH, config=CONFIG_PATH)
# Initiating camera
logger.info('Starting video stream...')

def uploadVideo1():
    global filename
    filename = filedialog.askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" \nvideo loaded from local system \n Detected objects are:\n")
    vs = cv2.VideoCapture(filename)
    time.sleep(2.0)
    fps = FPS().start()
    while True:
    # Reading frames
        frame = vs.read()
        frame = frame if filename is None else frame[1] 
        if frame is None:
            break
        (height, width) = frame.shape[:2]
        frame = imutils.resize(frame, width=500)

        # Converting frames to blobs using mean standardization
        blob=cv2.dnn.blobFromImage(image=frame, size=NETWORK_INPUT_SIZE,
                                   scalefactor=NETWORK_SCALE_FACTOR, mean=(5.5, 5.5, 5.5), crop=False)
        # Passing blob through neural network
        ssd_net.setInput(blob)
        network_output = ssd_net.forward()
        # Looping over detections
        for detection in network_output[0, 0]:
            score = float(detection[2])
            class_index = np.int(detection[1])
            label = f'{labels[class_index]}: {score:.2%}'
            # Drawing likely detections
            if score > SCORE_THRESHOLD:
                left = np.int(detection[3] * width)
                top = np.int(detection[4] * height)
                right = np.int(detection[5] * width)
                bottom = np.int(detection[6] * height)
                x=int(left-left/1.9)
                y=int(top-top/1.9)
                a=int(right-right/1.55)
                b=int(bottom-bottom/1.98)
                cv2.rectangle(img=frame, rec=(x, y, a, b), color=COLORS[class_index],
                            thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(img=frame, text=label, org=(x, np.int(y*0.9)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=COLORS[class_index], thickness=2, lineType=cv2.LINE_AA)
                engine = pyttsx3.init()
                voices = engine.getProperty('voices') #getting details of current voice
                engine.setProperty('voice', voices[1].id)
                engine.say(label)
                engine.runAndWait()
                text.insert(END,label+"\n")

        cv2.imshow("Detector", frame)

        # Exit loop by pressing "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()

    fps.stop()
    logger.info(f'\nElapsed time: {fps.elapsed() :.2f}')
    logger.info(f' Approx. FPS: {fps.fps():.2f}')
    cv2.destroyAllWindows()

                        
def images():
    global filename
    filename = filedialog.askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" \nimage loaded from local system \n Detected objects are:\n")
    img1= cv2.imread(filename)
    height, width, channels= img1.shape
    img1=cv2.resize(img1, None, fx=0.4, fy=0.4)
    while True:
            # Converting frames to blobs using mean standardization
        blob = cv2.dnn.blobFromImage(image=img1, size=NETWORK_INPUT_SIZE,
                 scalefactor=NETWORK_SCALE_FACTOR, mean=(2.5, 2.5, 2.5), crop=False) 
        # Passing blob through neural network
        ssd_net.setInput(blob)
        network_output = ssd_net.forward()
         # Looping over detections
        for detection in network_output[0, 0]:
            score = float(detection[2])
            class_index = np.int(detection[1])
            #print(class_index)
            label = f'{labels[class_index]}: {score:.2%}'
            # Drawing likely detections
            if score > SCORE_THRESHOLD:
                left = np.int(detection[3] * width)
                top = np.int(detection[4] * height)
                right = np.int(detection[5] * width)
                bottom = np.int(detection[6] * height)
                x=int(left-left/1.7)
                y=int(top-top/1.7)
                a=int(right-right/1.55)
                b=int(bottom-bottom/1.65)
                cv2.rectangle(img=img1, rec=(x, y, a, b), color=COLORS[class_index],
                            thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(img=img1, text=label, org=(x, np.int(y*0.9)), thickness=2,
 			fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                           	color=COLORS[class_index], lineType=cv2.LINE_AA)
                engine = pyttsx3.init()
                voices = engine.getProperty('voices') #getting details of current voice
                engine.setProperty('voice', voices[1].id)
                engine.say(label)
                engine.runAndWait()
                text.insert(END,label+"\n")
        cv2.imshow("Detector", img1)
        # Exit loop by pressing "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webcamVideo(s):
    text.delete('1.0', END)
    text.insert(END,"WEBCAM LIVE VIDEO INPUT\n OBJECTS DETECTED ARE:\n")
    logger.info('Starting video stream...')
    vs = VideoStream(src=s).start()
    time.sleep(1.0)
    fps = FPS().start()

    while True:
        # Reading frames
        frame = vs.read()
        frame = imutils.resize(frame, width=700)
        height, width, channels = frame.shape

        # Converting frames to blobs using mean standardization
        blob = cv2.dnn.blobFromImage(image=frame,
                                    scalefactor=NETWORK_SCALE_FACTOR,
                                    size=NETWORK_INPUT_SIZE,
                                    mean=(127.5, 127.5, 127.5),
                                    crop=False)

        # Passing blob through neural network
        ssd_net.setInput(blob)
        network_output = ssd_net.forward()

        # Looping over detections
        for detection in network_output[0, 0]:
            score = float(detection[2])
            class_index = np.int(detection[1])
            #print(class_index)
            label = f'{labels[class_index]}: {score:.2%}'
            # Drawing likely detections
            if score > SCORE_THRESHOLD:
                left = np.int(detection[3] * width)
                top = np.int(detection[4] * height)
                right = np.int(detection[5] * width)
                bottom = np.int(detection[6] * height)

                cv2.rectangle(img=frame,
                            rec=(left, top, right, bottom),
                            color=COLORS[class_index],
                            thickness=4,
                            lineType=cv2.LINE_AA)

                cv2.putText(img=frame,
                            text=label,
                            org=(left, np.int(top*0.9)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=COLORS[class_index],
                            thickness=3,
                            lineType=cv2.LINE_AA)
                engine = pyttsx3.init()
                engine.say(label)
                engine.runAndWait()
                text.insert(END,label+"\n")

        cv2.imshow("Detector", frame)

        # Exit loop by pressing "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        fps.update()

    fps.stop()
    logger.info(f'\nElapsed time: {fps.elapsed() :.2f}')
    logger.info(f' Approx. FPS: {fps.fps():.2f}')
    cv2.destroyAllWindows()
    vs.stop()




def exit():
    main.destroy()

    
font = ('times',20, 'bold')
title = Label(main, text='An Object Detection And Recognition Framework For Visually Impaired using yolo')
title.config(bg='green', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Browse System Videos", command=uploadVideo1)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Browse images", command=images)
uploadButton.place(x=256,y=150)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')  
pathlabel.config(font=font1)           
pathlabel.place(x=456,y=150)

webcamButton = Button(main, text="Start Webcam Video Tracking", command=lambda: webcamVideo(0))
webcamButton.place(x=50,y=200)
webcamButton.config(font=font1)

webcamButton = Button(main, text="Start Externalcam Video Tracking", command=lambda: webcamVideo(1))
webcamButton.place(x=330,y=200)
webcamButton.config(font=font1)


exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=640,y=200)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
