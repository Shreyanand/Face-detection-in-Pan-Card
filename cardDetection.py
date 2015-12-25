import cv2
import numpy as np


def detect_and_warp_card(img):

    ## DETECT CONTOURS AD SELECT THE CARD
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    edges =cv2.Canny(thresh,0,200)
    ab,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #contours = sorted(contours,key=cv2.contourArea,reverse=True)
    #cv2.drawContours(im, contours, -1, (0,255,0), 3)

    contour = contours[-1]
    l = 0
    for i in contours:
    
        a = cv2.arcLength(i,False)
        
        if a > 800:
            rect = cv2.minAreaRect(i)
            ar = rect[1][1]/rect[1][0]
            angle = rect[2]
            if ar > 1.3 and ar < 1.7:
                contour = i
            elif ar>0.4 and ar <0.8:
                contour = i 
            
    ###### CONVERT SELECTED CARD CONTOUR TO A RECTANGLE #######
                
    cv2.drawContours(img, contour, -1, (0,255,0), 3)
    card = contour
    peri = cv2.arcLength(card,True)
    approx = cv2.approxPolyDP(card,0.02*peri,True)
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    #print angle
    cr = cv2.boxPoints(rect)
    r = np.int0(cr)
    #print approx
    #cv2.drawContours(img, [r], 0,(0,0,255),2)

    ###### AFFINE TRANSFORMATION AND SHOW CARD #####
    
    r = np.float32(r)
    #print r
    if ((r[0][1]-r[3][1])**2+(r[0][0]-r[3][0])**2) < ((r[0][1]-r[1][1])**2+(r[0][0]-r[1
                                                                                   ][0])**2) :
        h = np.array([ [509,329],[0,329],[0,0],[509,0] ],np.float32)
    else:
        #h = np.array([ [509,329],[0,329],[0,0],[509,0] ],np.float32)
        h = np.array([[0,329],[0,0],[509,0],[509,329]],np.float32)
    
    transform = cv2.getPerspectiveTransform(r,h)
    warp = cv2.warpPerspective(img,transform,(510,330))
    
##    cv2.rectangle(img,(int(r[0][0]),int(r[0][1])),(int(r[0][0])+2,int(r[0][1])+2),(255,0,0),2)
##    cv2.rectangle(img,(int(r[1][0]),int(r[1][1])),(int(r[1][0])+2,int(r[1][1])+2),(255,255,0),2)
##    cv2.rectangle(img,(int(r[2][0]),int(r[2][1])),(int(r[2][0])+2,int(r[2][1])+2),(0,0,255),2)
##    cv2.rectangle(img,(int(r[3][0]),int(r[3][1])),(int(r[3][0])+2,int(r[3][1])+2),(0,255,0),2)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('ID card image',warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warp


def facedetect(imageF):
    finalimg = imageF
    face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(finalimg, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(finalimg,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = finalimg[y:y+h, x:x+w]
    cv2.imshow('face photo image',roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def textdetect(imageT):
    x = 20
    y = 90
    w = 140
    h = 33
    cv2.rectangle(imageT,(x,y),(x+w,y+h),(0,0,255),2)
    x = 17
    y = 172
    w = 140
    h = 25
    cv2.rectangle(imageT,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('face,text image',imageT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
####    gray = cv2.cvtColor(imageT,cv2.COLOR_BGR2GRAY)
####    blur = cv2.GaussianBlur(gray,(5,5),0)
####    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
####    der,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
####
####    samples =  np.empty((0,100))
####    responses = []
####    keys = [i for i in range(48,58)]
####    h=0
####    for cnt in contours:
####        if cv2.contourArea(cnt)<35:
####            [x,y,w,h] = cv2.boundingRect(cnt)
####
####        if  h>15:
    

im = cv2.imread('C:\Users\shrey\Desktop\New folder\Images\image2.jpg') # change PATH here
warped_card = detect_and_warp_card(im)
facedetect(warped_card)
textdetect(warped_card)


