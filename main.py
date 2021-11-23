import cv2 as cv
import cv2.cv2

capture = cv.VideoCapture(0)

three_ds_cascade = cv.cv2.CascadeClassifier('updated_haar_images/classifier/cascade.xml')

print("click w to close the capture")
while True:
    check, frame = capture.read() #update frames
    frame = cv.flip(frame, 1) #flips the camera so that it acts like a mirror

    #print(check)
    #print(frame)
    """
    Camera but grayscaled
    gray = cv.cv2.cvtColor(frame,cv.cv2.COLOR_BGR2GRAY)
    dscascades = three_ds_cascade.detectMultiScale(gray,1.01, 7)
    """

    dscascades = three_ds_cascade.detectMultiScale(frame, 1.01, 7)

    for(x,y,w,h) in dscascades:
        #gray = cv.cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        frame = cv.cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #frame as video, x and y for the top left corner, x+w and y+h will get the bottom corner, colour blue and the line thickness
        cv.cv2.putText(frame, '3DS', (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)

    #cv.imshow("Capture", gray)
    cv.imshow("Capture",frame)

    #haar_images/classifier/cascade.xml

    #cv.waitKey(0)
    key=cv.waitKey(1)

    if key==ord('w'):
        break

cv.destroyAllWindows()

#end camera
capture.release()


#capture.set(cv.cv.CV_CAP_PROP_FRAME_WIDTH,800)
#capture.set(cv.cv.CV_CAP_PROP_FRAME_HEIGHT,450)
#https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-set

"""
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# https://www.e-consystems.com/blog/camera/how-to-access-cameras-using-opencv-with-python/
# https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
# https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
#