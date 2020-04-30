from KeyPointDetector import detector
D = detector.Detector(path = './weights_2.pth',device="cpu")

import cv2
face_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
APP_NAME = 'NeuroFace'
def detect(img,cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 2)
    for (x, y, w, h) in faces:
        gray_img = gray[y:y+h,x:x+w]
        X,Y = D.detect(gray_img)
        X,Y = X*h,Y*w
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255))
        for x_,y_ in zip(X,Y):
            cv2.circle(img,(int(x+x_),int(y+y_)),1,(0,255,255),1,-1)
    return img
i =0;
while True:
    _, img = cap.read()
    # cv2.imwrite('results/input/input{}.png'.format(i),img)
    img = detect(img,face_cascade)
    # cv2.imwrite('results/output/output{}.png'.format(i),img)
    cv2.imshow(APP_NAME, img)
    i+=1
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()