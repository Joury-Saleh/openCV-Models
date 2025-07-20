import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

def calculate_EAR(eye):
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])
    x1 = dist.euclidean(eye[0], eye[3])
    return (y1 + y2) / (2.0 * x1)

def classify_age(age_str):
    if age_str in ['(0-2)', '(4-6)', '(8-12)']:
        return 'Child'
    elif age_str == '(15-20)':
        return 'Teen'
    elif age_str in ['(25-32)', '(38-43)', '(48-53)']:
        return 'Adult'
    else:
        return 'old'

blink_thresh = 0.25  # خفض العتبة لزيادة الحساسية
succ_frame = 3       # عدد الإطارات المتتالية لاعتبار الرمش
count_frame = 0
blink_total = 0
last_state = False   # لتتبع حالة العين

# تحميل نماذج كشف الوجه والمعالم
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# تحميل نماذج العمر والجنس
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_BUCKETS = ['Male', 'Female']

# تحميل الفيديو
video_path = 'old-women.mp4'
cam = cv2.VideoCapture(video_path)

while True:
    ret, frame = cam.read()
    if not ret:
        break  # انتهاء الفيديو

    frame = imutils.resize(frame, width=640)
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        continue  # تأكد من أن الصورة صحيحة

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[L_start:L_end]
        right_eye = shape[R_start:R_end]
        avg_EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # تحليل العمر والجنس
        blob = cv2.dnn.blobFromImage(cv2.resize(face_roi, (227, 227)),
                                     1.0, (227, 227),
                                     (78.426, 87.768, 114.895),
                                     swapRB=False)

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_range = AGE_BUCKETS[age_preds[0].argmax()]
        age_group = classify_age(age_range)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_BUCKETS[gender_preds[0].argmax()]

        # عد الرمشات
        if avg_EAR < blink_thresh:
            count_frame += 1
            if count_frame == succ_frame:
                blink_total += 1
        else:
            count_frame = 0  # إعادة العد إذا فتحت العين

        # عرض المعلومات على الفيديو
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'EAR: {avg_EAR:.2f}', (x, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Blinks: {blink_total}', (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'{age_group}, {gender}', (x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Blink + Age + Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
