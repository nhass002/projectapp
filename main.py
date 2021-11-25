import cv2 as cv #this library needs to be imported for opencv
import cv2.cv2

capture = cv.VideoCapture(0) #video capturing saved into a variable

three_ds_cascade = cv.cv2.CascadeClassifier('updated_haar_images/classifier/cascade.xml') #finds the classifier in the path

print("click w to close the capture")
while True:
    check, frame = capture.read() #update frames
    frame = cv.flip(frame, 1) #flips the camera so that it acts like a mirror

    #print(check)
    #print(frame)
    """
    Camera but grayscaled, decided not to use this
    gray = cv.cv2.cvtColor(frame,cv.cv2.COLOR_BGR2GRAY)
    dscascades = three_ds_cascade.detectMultiScale(gray,1.01, 7)
    """

    dscascades = three_ds_cascade.detectMultiScale(frame, 1.01, 7) #holds the classifier mutl

    for(x,y,w,h) in dscascades:
        #gray = cv.cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #frame as video, x and y for the top left corner, x+w and y+h will get the bottom corner, colour blue and the line thickness
        cv.cv2.putText(frame, '3DS', (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
        #takes in the frame image as parameter, labels 3ds above the rectangle, y-2 would let it sit on the rectangle, font, font scaling, font colour and font thickness

    #cv.imshow("Capture", gray)
    cv.imshow("Capture",frame) #displays the frame on screen

    key=cv.waitKey(1) #collects key clicks from the keyboard

    if key==ord('w'): #if it is key w then the camera window will close and the program ends
        break #breaks out of loop if condition is met

cv.destroyAllWindows()

#end camera
capture.release()


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