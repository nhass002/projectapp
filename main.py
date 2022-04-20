import cv2 as cv #this library needs to be imported for opencv
import cv2.cv2
#import argparse
from PyQt5 import QtCore, QtGui, QtWidgets

#GUI CLASS BELOW
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QFrame, QFileDialog
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject, QThread
import sys
import numpy as np

classesFile = 'coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))
#print
"""
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

widthHeight = 320

confThreshold = 0.5
nms = 0.3
"""


#https://www.imagetracking.org.uk/2020/12/displaying-opencv-images-in-pyqt/ THIS IS ONLY FOR DISPLAYING IMAGES IN GUI
class Thread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray) #uses numpy to turn the matrix into an array
    """
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False 
    """
#test
    def run(self):
        # capture from web cam
        capture = cv2.VideoCapture(0)
        self.modelConfiguration = 'yolov3.cfg'
        self.modelWeights = 'yolov3.weights'

        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.widthHeight = 480

        self.confThreshold = 0.5
        self.nms = 0.3

        while True: #while self._running

            check, cv_img = capture.read() #update frames
            #cv_img = cv.flip(cv_img, 1)
            #if check:
            self.blob = cv2.dnn.blobFromImage(cv_img, 1 / 255.0, (self.widthHeight, self.widthHeight), [0, 0, 0], 1, crop=False)
            self.net.setInput(self.blob)

            self.layerNames = self.net.getLayerNames()

            self.outputNames = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.outputs = self.net.forward(self.outputNames)

            self.findObjects(self.outputs, cv_img)
            self.change_pixmap_signal.emit(cv_img)

            #cv2.imshow('image', cv_img)
            #cv2.waitKey(1)

    def findObjects(self,outputs, img):
        height, width, center = img.shape
        bound = []
        classIDs = []
        confidence = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]  # first 5 are removed
                classID = np.argmax(scores)  # max value
                conf = scores[classID]
                if conf > self.confThreshold:
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2)
                    bound.append([x, y, w, h])
                    classIDs.append(classID)
                    confidence.append(float(conf))

        # print(len(bound))
        indices = cv2.dnn.NMSBoxes(bound, confidence, self.confThreshold, self.nms)
        # print(indices)
        for i in indices:
            # i = i[0]
            box = bound[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f'{classNames[classIDs[i]].upper()} {int(confidence[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#source of code that is adapted from https://www.imagetracking.org.uk/2020/12/displaying-opencv-images-in-pyqt/
#source 2 https://github.com/docPhil99/opencvQtdemo/blob/master/staticLabel2.py
class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Recognition GUI")
        self.disply_width = 720 #This is to keep it at a 720x480 screen size until further changes
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.video_label = QLabel(self)
        #self.centralwidget = QtWidgets.QWidget(MainWindow)
        #self.centralwidget.setObjectName("centralwidget")
        # create a text label
        self.textLabel = QLabel('Menu')

        self.modelConfiguration = 'yolov3.cfg'
        self.modelWeights = 'yolov3.weights'

        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.widthHeight = 320

        self.confThreshold = 0.5
        self.nms = 0.3

        widget = QWidget()
        #add 2 frames here
        #self.imageframe = QFrame()
        #self.videoframe = QFrame()
        #self.imageframe.resize(300,640)
        #self.videoframe.resize(300,640)

        #self.imageframe.add(self.image_label)
        #self.videoframe.add(self.video_label)

        self.pushButton = QtWidgets.QPushButton(widget)
        #self.pushButton.setGeometry(QtCore.QRect(10, 10, 75, 23)) #uncomment this later
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Switch Mode")

        #-----------------
        self.mode = 0 # change mode manually at start here
        #---------------------


        #screenshot
        self.pushButton_2 = QtWidgets.QPushButton(widget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Screenshot")
        #upload image
        self.pushButton_3 = QtWidgets.QPushButton(widget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Upload Image")

        self.comboBox = QtWidgets.QComboBox(widget)
        #self.comboBox.setGeometry(QtCore.QRect(10, 40, 171, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Choose Input Device")

        #buttons that arent being used will be greyed out
        self.pushButton_2.setEnabled(False)
        #self.pushButton_3.setEnabled(False)

        self.pushButton.clicked.connect(self.switchMode)

        self.pushButton_3.clicked.connect(self.upload_image)

        # create a vertical box layout and add the two labels, then add buttons and combobox (drop down menu)
        vbox = QVBoxLayout()
        #vbox.addWidget(self.imageframe)
        #vbox.addWidget(self.videoframe)
        vbox.addWidget(self.image_label) #label for displaying the image within
        vbox.addWidget(self.video_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.pushButton) #switch modes
        vbox.addWidget(self.pushButton_2) #screenshot
        vbox.addWidget(self.pushButton_3) #upload image
        vbox.addWidget(self.comboBox) #this will be used for possibly changing the camera
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        self.image_label.hide()
        self.video_label.hide()

        if self.mode == 0: #IN MODE 0 IT WILL DISPLAY AN IMAGE
            self.image_label.show()
            # create a grey pixmap
            grey = QPixmap(self.disply_width, self.display_height)
            grey.fill(QColor('darkGray'))
            # set the image image to the grey pixmap
            self.image_label.setPixmap(grey)
            self.img = cv.imread("updated_haar_images/test_files_grayscale/apple_80.jpg")

            #
            self.blob = cv2.dnn.blobFromImage(self.img, 1 / 255.0, (self.widthHeight, self.widthHeight), [0, 0, 0], 1,
                                              crop=False)
            self.net.setInput(self.blob)

            self.layerNames = self.net.getLayerNames()

            self.outputNames = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.outputs = self.net.forward(self.outputNames)


            # perform detection on the image
            #self.detect(self.img)
            self.findObjects(self.outputs, self.img)
            # convert the image to Qt format
            qt_img = self.convert_cv_qt(self.img)
            # display it
            self.image_label.setPixmap(qt_img)

        if self.mode == 1: #IN MODE 1 IT WILL DISPLAY THE LIVE CAMERA
            self.video_label.show()
            # create the video capture thread
            self.thread = Thread()
            # connect its signal to the update_image slot
            self.thread.change_pixmap_signal.connect(self.update_image)
            # start the thread
            self.thread.start()

    def switchMode(self):
        if self.mode == 1: # switches from video to image
            self.mode = 0
            print(self.mode)
            self.image_label.show()
            self.video_label.hide()
            #self.thread.capture.release()
            #self.thread.time.sleep(0.1)
            # NEW ##
            #"""
            # create a grey pixmap
            grey = QPixmap(self.disply_width, self.display_height)
            grey.fill(QColor('darkGray'))
            # set the image image to the grey pixmap
            self.image_label.setPixmap(grey)
            self.img = cv.imread("updated_haar_images/test_files_grayscale/apple_80.jpg")
            #
            self.blob = cv2.dnn.blobFromImage(self.img, 1 / 255.0, (self.widthHeight, self.widthHeight), [0, 0, 0], 1,
                                              crop=False)
            self.net.setInput(self.blob)

            self.layerNames = self.net.getLayerNames()

            self.outputNames = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.outputs = self.net.forward(self.outputNames)


            # perform detection on the image
            #self.detect(self.img)
            self.findObjects(self.outputs, self.img)
            # convert the image to Qt format
            qt_img = self.convert_cv_qt(self.img)
            # display it
            self.image_label.setPixmap(qt_img)
            #self.thread.stop()
            #self.thread.terminate()
            #"""
        elif self.mode == 0:
            self.mode = 1; # switches from image to video
            print(self.mode)
            self.video_label.show()
            self.image_label.hide()
            ## NEWLY ADDED ##
            #"""
            # create the video capture thread
            self.thread = Thread() #target = self.thread.run
            # connect its signal to the update_image slot
            self.thread.change_pixmap_signal.connect(self.update_image)
            # start the thread
            self.thread.start()
            #"""
            """
            capture.release()
            #self.thread.quit()
            img = cv.imread("updated_haar_images/test_files_grayscale/apple_80.jpg")
            # perform detection on the image
            self.detect(img)
            # convert the image to Qt format
            qt_img = self.convert_cv_qt(img)
            # display it
            self.image_label.setPixmap(qt_img)
            """

    #function for uploading image button
    @pyqtSlot()
    def upload_image(self):
        #print("e")
        self.filename = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\')
        self.path = self.filename[0]

        #print(self.filename)
        #print(self.path)

        grey = QPixmap(self.disply_width, self.display_height)
        grey.fill(QColor('darkGray'))
        # set the image to the grey pixmap
        self.image_label.setPixmap(grey)
        # read image file here
        self.img = cv.imread(self.path) #self.file_image

        #
        self.blob = cv2.dnn.blobFromImage(self.img, 1 / 255.0, (self.widthHeight, self.widthHeight), [0, 0, 0], 1,
                                          crop=False)
        self.net.setInput(self.blob)

        self.layerNames = self.net.getLayerNames()

        self.outputNames = [self.layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.outputs = self.net.forward(self.outputNames)

        # perform detection on the image
        # self.detect(self.img)
        self.findObjects(self.outputs, self.img)
        # convert the image to Qt format
        qt_img = self.convert_cv_qt(self.img)
        # display it
        self.image_label.setPixmap(qt_img)

    #function for screenshotting images

    @pyqtSlot(np.ndarray) #converts python method into a qt slot for a signal which is connected earlier
    def update_image(self, cv_img):
        #Updates the image_label with a new opencv image
        qt_img = self.convert_cv_qt(cv_img)
        if self.mode == 0: #image
            self.image_label.setPixmap(qt_img)
        elif self.mode == 1: #video
            self.video_label.setPixmap(qt_img)
        #detect(qt_img)
        #self.findObjects(self.outputs, self.img)

    def convert_cv_qt(self, cv_img):
        #Convert from an opencv image to QPixmap
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def findObjects(self,outputs, img):
        height, width, center = img.shape
        bound = []
        classIDs = []
        confidence = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]  # first 5 are removed
                classID = np.argmax(scores)  # max value
                conf = scores[classID]
                if conf > self.confThreshold:
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2)
                    bound.append([x, y, w, h])
                    classIDs.append(classID)
                    confidence.append(float(conf))

        # print(len(bound))
        indices = cv2.dnn.NMSBoxes(bound, confidence, self.confThreshold, self.nms)
        # print(indices)
        for i in indices:
            # i = i[0]
            box = bound[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f'{classNames[classIDs[i]].upper()} {int(confidence[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    """
    def detect(self, frame):
        #dscascades = self.three_ds_cascade.detectMultiScale(frame, 1.01,
                                                            #7)  # holds the classifier multiscale which does the detecting
        # and returns the boundaries for the rectangle
        # fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)
        fruitcascades = self.fruits_cascade.detectMultiScale(frame, 1.01, 7)
        applecascades = self.apples_cascade.detectMultiScale(frame, 1.01, 7)
        # bananacascades = bananas_cascade.detectMultiScale(frame, 1.01, 7)

        for (x, y, w, h) in fruitcascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'fruits', ((x + w) - 10, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        for (x, y, w, h) in applecascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'apple', ((x + w) - 30, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        # for (x, y, w, h) in bananacascades:
        # frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv.cv2.putText(frame, 'banana', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)

        #Source: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

        # -- Detect faces
        face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'face', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)


#capture = cv.VideoCapture(0) #video capturing saved into a variable


three_ds_cascade = cv.cv2.CascadeClassifier('updated_haar_images/classifier/cascade.xml') #finds the classifier in the path

fruits_cascade = cv.cv2.CascadeClassifier('updated_haar_images/fruitcascade.xml')
apples_cascade = cv.cv2.CascadeClassifier('updated_haar_images/applecascade.xml')
bananas_cascade = cv.cv2.CascadeClassifier('updated_haar_images/bananacascade.xml')

#mode = 1;
    """


if __name__ == "__main__":
    import sys

    mode = 1

    #app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    #ui = Ui_MainWindow()
    app = QApplication(sys.argv)
    a = mainWindow()
    a.show()
    sys.exit(app.exec_())

print("click w to close the capture")
print("click s to switch to load image")
print("click c to switch back to camera")
while True:

    #ui.setupUi(MainWindow)
    #ui.buttonCheck(mode)  # recently added
    #ui.display(mode)
    #MainWindow.show()

    """
    def display(mode):
        if mode == 0:
            img = cv.imread("updated_haar_images/test_files_grayscale/apple_80.jpg")  # apple_78
            # img = cv.imread("updated_haar_images/test/mixed_6.jpg")
            #gray = cv.cv2.cvtColor(img, cv.cv2.COLOR_BGR2GRAY)
            windowname = "Image Display"
            detect(img)
        if mode == 1:
            check, frame = capture.read() #update frames
            frame = cv.flip(frame, 1) #flips the camera so that it acts like a mirror
            #windowcapture = "Capture"
            detect(frame)
    

    def detect(frame):
        dscascades = three_ds_cascade.detectMultiScale(frame, 1.01,
                                                       7)  # holds the classifier multiscale which does the detecting
        # and returns the boundaries for the rectangle
        # fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)
        fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)
        applecascades = apples_cascade.detectMultiScale(frame, 1.01, 7)
        bananacascades = bananas_cascade.detectMultiScale(frame, 1.01, 7)
        
        for (x, y, w, h) in dscascades:
            # gray = cv.cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # frame as video, x and y for the top left corner, x+w and y+h will get the bottom corner, colour blue and the line thickness
            cv.cv2.putText(frame, '3DS', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            # takes in the frame image as parameter, labels 3ds above the rectangle, y-2 would let it sit on the rectangle, font, font scaling, font colour and font thickness
        
        for (x, y, w, h) in fruitcascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'fruits', ((x + w) - 10, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        for (x, y, w, h) in applecascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'apple', ((x + w) - 30, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        for (x, y, w, h) in bananacascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'banana', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)

            # for (x, y, w, h) in fruitcascades:
            # frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv.cv2.putText(frame, 'fruits', ((x + w) - 10, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            # for (x, y, w, h) in applecascades:
            # frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv.cv2.putText(frame, 'apple', ((x + w) - 30, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            # for (x, y, w, h) in bananacascades:
            # frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv.cv2.putText(frame, 'banana', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)

        # -- Detect faces
        face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'face', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
    
    if mode == 0:
        img = cv.imread("updated_haar_images/test_files_grayscale/apple_80.jpg") #apple_78
        #img = cv.imread("updated_haar_images/test/mixed_6.jpg")
        gray = cv.cv2.cvtColor(img, cv.cv2.COLOR_BGR2GRAY)
        windowname = "Image Display"
        ui.detect(img)
        cv.imshow(windowname, img)  # "Display window"


    if mode == 1:
        ui.display()
        check, frame = capture.read() #update frames
        frame = cv.flip(frame, 1) #flips the camera so that it acts like a mirror
        windowcapture = "Capture"
        ui.detect(frame)
    

    def greyscale(frame):
            #Camera but grayscaled, decided not to use this yet
        gray = cv.cv2.cvtColor(frame, cv.cv2.COLOR_BGR2GRAY)
        dscascades = three_ds_cascade.detectMultiScale(gray, 1.01, 7)

        #cv.imshow("Capture", gray)
        #cv.imshow(windowcapture,frame) #displays the frame on screen

    key=cv.waitKey(1) #collects key clicks from the keyboard

    if key==ord('w'): #if it is key w then the camera window will close and the program ends
        break #breaks out of loop if condition is met
    if key==ord('s'):
        cv.destroyWindow() #windowcapture
        mode = 0
    if key==ord('c'):
        cv.destroyWindow() #windowname
        mode = 1

cv.destroyAllWindows()

#end camera
#capture.release()
sys.exit(mainWindow.exec_())


#if __name__ == "__main__":
    #import sys
    #app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    #ui = Ui_MainWindow()
    #ui.setupUi(MainWindow)
    #MainWindow.show()
    #sys.exit(app.exec_())
    
"""



#capture.set(cv.cv.CV_CAP_PROP_FRAME_WIDTH,800)
#capture.set(cv.cv.CV_CAP_PROP_FRAME_HEIGHT,450)
#https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set

#extra info below

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# https://www.e-consystems.com/blog/camera/how-to-access-cameras-using-opencv-with-python/
# https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
#