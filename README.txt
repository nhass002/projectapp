opencv needs to be pip installed
pyqt5 needs to be installed
numpy needs to be installed

download the weights file and keep it with main.py https://drive.google.com/file/d/1YqQx4CGi1ijba7KBwqz0Efnb2wePNOEi/view?usp=sharing
or it can be downloaded here as yolov3-320 weights file https://pjreddie.com/darknet/yolo/

main.py is the main file, it needs coco.names, yolov3.cfg AND the weights file above. The test images can be downloaded however they're optional. The updated_haar_images do not need to be downloaded however they're just proof for what I used to train the cascade classifiers.

1. Run on code
2. Click the Switch Mode button and hold up objects to the camera or just use recognition on your face/body
- If the mode is currently displaying an image, the Upload Button can be clicked and it will perform detection on the image
- Upload image button does not work on the video capture mode intentionally.
- If Switch Mode is clicked while the video capture is on, it will stop the video capture on the last frame. So the upload image button will work.
3. Click X on the window to end the program
