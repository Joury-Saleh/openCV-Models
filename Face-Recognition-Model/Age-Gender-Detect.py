import cv2
import os
from pathlib import Path

MODEL_MEAN_VALUES = (78.4463377603,
                     87.7689143744,
                     114.895847746)

age_list = ['(0,2)','(4-6)','(8-12)','(15-20)','(25-32)','(35-43)','(45-53)','(60-100)']
gender_list = ['Male','Female']

def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel'
        
    )

    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )
    return(age_net,gender_net)

def read_from_img(age_net,gender_net):
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    img = cv2.imread('images/girl1.jpg')
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,5)

    if(len(faces) > 0):
        print("Found {} Faces".format(str(len(faces))))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),( x+w , y+h ),(255,255,0),2)
        face_img = img[y:y+h , h:h+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (250,250),MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob) 
        gender_p = gender_net.forward()
        gender = gender_list[gender_p[0].argmax()]
        print("Gender: "+gender)   
        
        age_net.setInput(blob) 
        age_p = age_net.forward()
        age = age_list[age_p[0].argmax()]
        print("Age: "+age)   

        G_A = "%s %s" % (gender , age)
        cv2.putText(img, G_A , (x,y) , font , 1 , (255,255,255) , 2 ,cv2.LINE_AA)
        cv2.imshow("Face-Detect" , img)

    k = cv2.waitKey(0)

    if k == 27 :
        cv2.destroyAllWindows()
    elif k == ord('s'):
        downloads_path = str(Path.home() / "Downloads")
        save_path = os.path.join(downloads_path, "detected_output.png")
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    age_net, gender_net = filesGet()
    read_from_img(age_net,gender_net)