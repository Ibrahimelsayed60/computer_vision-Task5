
import cv2
face = cv2.CascadeClassifier('cascade\haarcascade_frontalface_alt2.xml')
# img = cv2.imread('data\image_050.jpg')
def facedetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    faces = face.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h)in faces:
        roi = gray[y: y+h, x:x+w]
        color = (255,0,0)
        stroke = 2
        width = x+w
        height = y+h
        cv2.rectangle(img,(x,y),(width, height),color, stroke)
    cv2.imwrite('img.png',img)
# cv2.waitKey(0)