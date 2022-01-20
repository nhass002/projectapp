import cv2 as cv #this library needs to be imported for opencv
import cv2.cv2
from PyQt5 import QtCore, QtGui, QtWidgets

capture = cv.VideoCapture(0) #video capturing saved into a variable

three_ds_cascade = cv.cv2.CascadeClassifier('updated_haar_images/classifier/cascade.xml') #finds the classifier in the path

fruits_cascade = cv.cv2.CascadeClassifier('updated_haar_images/fruitcascade.xml')
apples_cascade = cv.cv2.CascadeClassifier('updated_haar_images/applecascade.xml')
bananas_cascade = cv.cv2.CascadeClassifier('updated_haar_images/bananacascade.xml')

mode = 1;

print("click w to close the capture")
while True:

    if mode == 0:
        img = cv.imread("updated_haar_images/test/apple_78.jpg") #apple_79 #mixed_6
        #img = cv.imread("updated_haar_images/test/mixed_6.jpg")
        gray = cv.cv2.cvtColor(img, cv.cv2.COLOR_BGR2GRAY)
        windowname = "Image Display"
        fruitcascades = fruits_cascade.detectMultiScale(img, 1.01, 7)
        applecascades = apples_cascade.detectMultiScale(img, 1.01, 7)
        bananacascades = bananas_cascade.detectMultiScale(img, 1.01, 7)
        #detect()
        for (x,y,w,h) in fruitcascades:
            img = cv.cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(img, 'fruits', ((x + w)-30, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        for (x,y,w,h) in applecascades:
            img = cv.cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(img, 'apple', ((x + w)-30, (y+h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        for (x,y,w,h) in bananacascades:
            img = cv.cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(img, 'banana', (x, (y+h) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        cv.imshow(windowname, img)  # "Display window"

    if mode == 1:
        check, frame = capture.read() #update frames
        frame = cv.flip(frame, 1) #flips the camera so that it acts like a mirror
        windowcapture = "Capture"
        #detect()

        #print(check)
        #print(frame)
    #def detect():
        def greyscale():
            #Camera but grayscaled, decided not to use this yet
            gray = cv.cv2.cvtColor(frame, cv.cv2.COLOR_BGR2GRAY)
            dscascades = three_ds_cascade.detectMultiScale(gray, 1.01, 7)

        dscascades = three_ds_cascade.detectMultiScale(frame, 1.01, 7) #holds the classifier multiscale which does the detecting
        # and returns the boundaries for the rectangle
        fruitcascades = fruits_cascade.detectMultiScale(frame, 1.01, 7)

        for(x,y,w,h) in dscascades:
            #gray = cv.cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #frame as video, x and y for the top left corner, x+w and y+h will get the bottom corner, colour blue and the line thickness
            cv.cv2.putText(frame, '3DS', (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            #takes in the frame image as parameter, labels 3ds above the rectangle, y-2 would let it sit on the rectangle, font, font scaling, font colour and font thickness

        for (x,y,w,h) in fruitcascades:
            frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.cv2.putText(frame, 'fruits', ((x + w)-10, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        #cv.imshow("Capture", gray)
        cv.imshow(windowcapture,frame) #displays the frame on screen

    key=cv.waitKey(1) #collects key clicks from the keyboard

    if key==ord('w'): #if it is key w then the camera window will close and the program ends
        break #breaks out of loop if condition is met
    if key==ord('s'):
        cv.destroyWindow(windowcapture)
        mode = 0
    if key==ord('c'):
        cv.destroyWindow(windowname)
        mode = 1

cv.destroyAllWindows()

#end camera
capture.release()
#GUI CLASS BELOW
"""
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Default Input"))
        self.pushButton_3.setText(_translate("MainWindow", "Screenshot"))
        self.pushButton.setText(_translate("MainWindow", "Switch Mode"))
        self.pushButton_4.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_2.setText(_translate("MainWindow", "Upload Image"))


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