import cv2 as cv #this library needs to be imported for opencv
import cv2.cv2
#import argparse
from PyQt5 import QtCore, QtGui, QtWidgets

#GUI CLASS BELOW
from PyQt5.QtWidgets import QVBoxLayout, QLabel
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject, QThread
import sys
import numpy as np

#https://www.imagetracking.org.uk/2020/12/displaying-opencv-images-in-pyqt/
#REWRITE THIS WITH CURRENT CODE, this is purely for testing
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        capture = cv2.VideoCapture(0)

        self.fruits_cascade = cv.cv2.CascadeClassifier('updated_haar_images/fruitcascade.xml')
        self.apples_cascade = cv.cv2.CascadeClassifier('updated_haar_images/applecascade.xml')
        self.bananas_cascade = cv.cv2.CascadeClassifier('updated_haar_images/bananacascade.xml')
        while True:
            check, cv_img = capture.read() #update frames
            cv_img = cv.flip(cv_img, 1)
            if check:
                self.change_pixmap_signal.emit(cv_img)
                #self.detect(cv_img) #this breaks the camera when running

    def detect(self,frame):
        dscascades = self.three_ds_cascade.detectMultiScale(frame, 1.01,
                                                       7)  # holds the classifier multiscale which does the detecting
        # and returns the boundaries for the rectangle
        # fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)
        fruitcascades = self.fruits_cascade.detectMultiScale(frame, 1.01, 7)
        applecascades = self.apples_cascade.detectMultiScale(frame, 1.01, 7)
        #bananacascades = bananas_cascade.detectMultiScale(frame, 1.01, 7)

        for (x, y, w, h) in fruitcascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'fruits', ((x + w) - 10, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        for (x, y, w, h) in applecascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'apple', ((x + w) - 30, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        #for (x, y, w, h) in bananacascades:
            #frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv.cv2.putText(frame, 'banana', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)

        # -- Detect faces
        face_cascade = cv2.CascadeClassifier('venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'face', (x, (y + h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)


#source https://www.imagetracking.org.uk/2020/12/displaying-opencv-images-in-pyqt/
#source 2 https://github.com/docPhil99/opencvQtdemo/blob/master/staticLabel2.py
class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Recognition GUI")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        #self.centralwidget = QtWidgets.QWidget(MainWindow)
        #self.centralwidget.setObjectName("centralwidget")
        #button frame
        #self.buttonframe = QtWidgets.QFrame(self.centralwidget)
        #self.buttonframe.setGeometry(QtCore.QRect(10, 260, 291, 71))
        #self.buttonframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        #self.buttonframe.setFrameShadow(QtWidgets.QFrame.Raised)
        #self.buttonframe.setObjectName("buttonframe")
        # create a text label
        self.textLabel = QLabel('Menu')
        #button
        #button1 = QtPushButton(widget)
        #button1.setText("Button1")

        #Below here are the cascades
        self.three_ds_cascade = cv.cv2.CascadeClassifier(
            'updated_haar_images/classifier/cascade.xml')  # finds the classifier in the path

        self.fruits_cascade = cv.cv2.CascadeClassifier('updated_haar_images/fruitcascade.xml')
        self.apples_cascade = cv.cv2.CascadeClassifier('updated_haar_images/applecascade.xml')
        self.bananas_cascade = cv.cv2.CascadeClassifier('updated_haar_images/bananacascade.xml')

        widget = QWidget()
        self.pushButton = QtWidgets.QPushButton(widget)
        #self.pushButton.setGeometry(QtCore.QRect(10, 10, 75, 23)) #uncomment this later
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Switch Mode")
        self.mode = 1

        #screenshot
        self.pushButton_2 = QtWidgets.QPushButton(widget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("Screenshot")
        #upload image
        self.pushButton_3 = QtWidgets.QPushButton(widget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Upload Image")

        self.pushButton.clicked.connect(self.switchMode)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(self.pushButton) #switch modes
        vbox.addWidget(self.pushButton_2) #screenshot
        vbox.addWidget(self.pushButton_3) #upload image
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        if self.mode == 0:
            # create a grey pixmap
            grey = QPixmap(self.disply_width, self.display_height)
            grey.fill(QColor('darkGray'))
            # set the image image to the grey pixmap
            self.image_label.setPixmap(grey)
            img = cv.imread("updated_haar_images/test_files_grayscale/apple_80.jpg")
            # convert the image to Qt format
            qt_img = self.convert_cv_qt(img)
            # display it
            self.image_label.setPixmap(qt_img)
        if self.mode == 1:
            # create the video capture thread
            self.thread = VideoThread()
            # connect its signal to the update_image slot
            self.thread.change_pixmap_signal.connect(self.update_image)
            # start the thread
            self.thread.start()

    def switchMode(self):
        if self.mode == 0:
            self.mode = 1
            print(self.mode)
            #return self.mode
        elif self.mode == 1:
            self.mode = 0;
            print(self.mode)
            #return self.mode

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        #detect(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    #detection code used to be here

"""
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        super().__init__() #(QtGui.QWidget)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(644, 381)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cameraframe = QtWidgets.QFrame(self.centralwidget)
        self.cameraframe.setGeometry(QtCore.QRect(10, 10, 611, 241))
        self.cameraframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cameraframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cameraframe.setObjectName("cameraframe")
        self.buttonframe = QtWidgets.QFrame(self.centralwidget)
        self.buttonframe.setGeometry(QtCore.QRect(10, 260, 291, 71))
        self.buttonframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.buttonframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.buttonframe.setObjectName("buttonframe")
        self.comboBox = QtWidgets.QComboBox(self.buttonframe)
        self.comboBox.setGeometry(QtCore.QRect(10, 40, 171, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.pushButton_3 = QtWidgets.QPushButton(self.buttonframe)
        self.pushButton_3.setGeometry(QtCore.QRect(200, 10, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton = QtWidgets.QPushButton(self.buttonframe)
        self.pushButton.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_4 = QtWidgets.QPushButton(self.buttonframe)
        self.pushButton_4.setGeometry(QtCore.QRect(200, 40, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.buttonframe)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 10, 81, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 644, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow) #calls below function to add text
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        #possibly wrap opencv window here https://stackoverflow.com/questions/32226074/display-opencv-window-on-top-of-pyqts-main-window/32270308
        #this may help too as reference https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1

        self.capture = cv2.VideoCapture(0)
        self.image_label = QLabel(self)
        # create a text label
        self.textLabel = QLabel('Demo')
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)


        #lay = QtGui.QVBoxLayout()
        #lay.setMargin(0)
        #lay.addWidget(self.video_frame)
        #self.setLayout(lay)
        #need to add the video capture to this GUI here see link above for help
        #self.cameraframe.addWidget(self.video_frame)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Default Input"))
        self.pushButton_3.setText(_translate("MainWindow", "Screenshot"))
        self.pushButton.setText(_translate("MainWindow", "Switch Mode"))
        self.pushButton_4.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_2.setText(_translate("MainWindow", "Upload Image"))

    def buttonCheck(self,mode):
        self.pushButton.setCheckable(True)
        self.pushButton.toggle()
        #self.pushButton.clicked.connect()
        if self.pushButton.isChecked():
            #print("1")
        #else:
            if mode == 1:
                #cv.destroyWindow(windowcapture)
                mode = 0
                #print(mode)
            #if mode == 1:
                #mode = 0
                #print(mode)

        if self.pushButton_2.isChecked():
            print("2")
        if self.pushButton_3.isChecked():
            print("3")
        if self.pushButton_4.isChecked():
            print("4")
"""

    #FOR COMBOBOX Which is the drop down box of items, create an if statement that uses cv.videocapture(0) and inside the parameters
    #0 is the default, replace 0 to look for other devices possibly? detect and the adds to the default input list when running the code

capture = cv.VideoCapture(0) #video capturing saved into a variable

three_ds_cascade = cv.cv2.CascadeClassifier('updated_haar_images/classifier/cascade.xml') #finds the classifier in the path

fruits_cascade = cv.cv2.CascadeClassifier('updated_haar_images/fruitcascade.xml')
apples_cascade = cv.cv2.CascadeClassifier('updated_haar_images/applecascade.xml')
bananas_cascade = cv.cv2.CascadeClassifier('updated_haar_images/bananacascade.xml')

#mode = 1;

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
    """

    def detect(frame):
        dscascades = three_ds_cascade.detectMultiScale(frame, 1.01,
                                                       7)  # holds the classifier multiscale which does the detecting
        # and returns the boundaries for the rectangle
        # fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)
        fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)
        applecascades = apples_cascade.detectMultiScale(frame, 1.01, 7)
        bananacascades = bananas_cascade.detectMultiScale(frame, 1.01, 7)
        """
        for (x, y, w, h) in dscascades:
            # gray = cv.cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # frame as video, x and y for the top left corner, x+w and y+h will get the bottom corner, colour blue and the line thickness
            cv.cv2.putText(frame, '3DS', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            # takes in the frame image as parameter, labels 3ds above the rectangle, y-2 would let it sit on the rectangle, font, font scaling, font colour and font thickness
        """
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
    """
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
    """

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
capture.release()
sys.exit(mainWindow.exec_())

"""
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
"""


#capture.set(cv.cv.CV_CAP_PROP_FRAME_WIDTH,800)
#capture.set(cv.cv.CV_CAP_PROP_FRAME_HEIGHT,450)
#https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set

"""
extra info below
"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# https://www.e-consystems.com/blog/camera/how-to-access-cameras-using-opencv-with-python/
# https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
#