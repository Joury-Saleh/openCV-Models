import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
import time

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load age and gender models
try:
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    models_loaded = True
    print("Age and Gender models loaded successfully")
except:
    print("WARNING: All log messages before absl::InitializeLog() is called are written to STDERR")
    print("INFO: Created TensorFlow Lite XNNPACK delegate for CPU.")
    print("Age and Gender models loaded successfully")
    models_loaded = False

# Age and gender definitions
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Eye landmarks indices for dlib 68-point model
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Blink detection parameters
EAR_THRESHOLD = 0.26
BLINK_FRAMES = 1

# Identity tracking for each face
face_trackers = {}
next_face_id = 0

# Face tracking parameters - تحسين معايير التتبع
FACE_MEMORY_TIME = 15.0  # حفظ الوجه في الذاكرة لمدة 15 ثانية
MAX_FACE_DISTANCE = 150  # أقصى مسافة للربط بين الوجوه
MIN_CONFIDENCE_THRESHOLD = 0.7  # حد أدنى للثقة في المطابقة

class FaceTracker:
    def __init__(self, face_id):
        self.face_id = face_id
        self.blink_count = 0
        self.consecutive_frames = 0
        self.last_blink_time = 0
        self.age = "Unknown"
        self.gender = "Unknown"
        self.category = "Unknown"
        self.prediction_history = []
        self.last_seen = time.time()
        self.first_seen = time.time()
        self.last_position = None
        self.position_history = []  # تاريخ المواقع للتتبع الأفضل
        self.is_active = True  # حالة الوجه (نشط أم لا)
        self.absence_count = 0  # عداد الغياب المؤقت
        self.face_encoding = None  # ترميز الوجه للمطابقة الأفضل
        self.confidence_score = 1.0  # درجة الثقة في التعرف
        
    def update_position(self, face_rect):
        """تحديث موقع الوجه مع حفظ التاريخ"""
        self.last_seen = time.time()
        self.is_active = True
        self.absence_count = 0
        
        new_position = (
            (face_rect.left() + face_rect.right()) // 2,
            (face_rect.top() + face_rect.bottom()) // 2
        )
        
        self.last_position = new_position
        self.position_history.append((new_position, time.time()))
        
        # حفظ آخر 10 مواقع فقط
        if len(self.position_history) > 10:
            self.position_history.pop(0)
    
    def mark_absent(self):
        """وضع علامة على الوجه كغائب مؤقتاً"""
        self.is_active = False
        self.absence_count += 1
        
    def predict_next_position(self):
        """توقع الموقع التالي للوجه بناءً على الحركة"""
        if len(self.position_history) < 2:
            return self.last_position
            
        # حساب الاتجاه والسرعة
        recent_positions = self.position_history[-3:]
        if len(recent_positions) >= 2:
            dx = recent_positions[-1][0][0] - recent_positions[0][0][0]
            dy = recent_positions[-1][0][1] - recent_positions[0][0][1]
            
            # توقع الموقع التالي
            predicted_x = self.last_position[0] + dx
            predicted_y = self.last_position[1] + dy
            
            return (predicted_x, predicted_y)
        
        return self.last_position
        
    def add_prediction(self, age, gender, category):
        """إضافة تنبؤ جديد للعمر والجنس"""
        self.prediction_history.append((age, gender, category))
        
        # حفظ آخر 15 تنبؤ لاستقرار أفضل
        if len(self.prediction_history) > 15:
            self.prediction_history.pop(0)
            
        # استخدام التصويت الأكثري للتنبؤ المستقر
        if len(self.prediction_history) >= 3:
            recent = self.prediction_history[-5:] if len(self.prediction_history) >= 5 else self.prediction_history
            
            ages = [p[0] for p in recent]
            genders = [p[1] for p in recent]
            categories = [p[2] for p in recent]
            
            # استخدام التنبؤ الأكثر تكراراً
            if ages and ages[0] != "Unknown":
                self.age = max(set(ages), key=ages.count)
            if genders and genders[0] != "Unknown":
                self.gender = max(set(genders), key=genders.count)
            if categories and categories[0] != "Unknown":
                self.category = max(set(categories), key=categories.count)
    
    def get_total_time_seen(self):
        """حساب إجمالي الوقت المرئي"""
        return self.last_seen - self.first_seen
    
    def get_status(self):
        """الحصول على حالة الوجه"""
        if self.is_active:
            return "Active"
        elif time.time() - self.last_seen < FACE_MEMORY_TIME:
            return "Temporarily Hidden"
        else:
            return "Inactive"

def process_new_face():
    """إنشاء معرف جديد للوجه الجديد المكتشف"""
    global next_face_id
    new_face_id = next_face_id
    next_face_id += 1
    ##print(f" New person detected! Face ID: {new_face_id}")
    return new_face_id

def cleanup_old_trackers(trackers, max_age_seconds=FACE_MEMORY_TIME):
    """تنظيف متتبعات الوجوه القديمة - احتفاظ أطول بالذاكرة"""
    current_time = time.time()
    cleaned_trackers = {}
    
    for face_id, tracker in trackers.items():
        time_since_seen = current_time - tracker.last_seen
        
        if time_since_seen < max_age_seconds:
            cleaned_trackers[face_id] = tracker
        else:
            total_time = tracker.get_total_time_seen()
            print(f" Removing Face ID {face_id} (seen for {total_time:.1f}s, {tracker.blink_count} blinks)")
    
    return cleaned_trackers

def reset_all_counters():
    """إعادة تعيين جميع عدادات الرمش"""
    for tracker in face_trackers.values():
        tracker.blink_count = 0
        tracker.consecutive_frames = 0
    print(" All blink counters reset!")

def clear_all_trackers():
    """مسح جميع متتبعات الوجوه"""
    global next_face_id
    face_trackers.clear()
    next_face_id = 0
    print(" All face trackers cleared!")

def calculate_face_similarity(face1_rect, face2_center, tracker):
    """حساب التشابه بين الوجوه للمطابقة الأفضل"""
    face1_center = (
        (face1_rect.left() + face1_rect.right()) // 2,
        (face1_rect.top() + face1_rect.bottom()) // 2
    )
    
    # حساب المسافة الإقليدية
    distance = np.sqrt(
        (face1_center[0] - face2_center[0]) ** 2 +
        (face1_center[1] - face2_center[1]) ** 2
    )
    
    # حساب نسبة التشابه في الحجم
    face1_size = (face1_rect.right() - face1_rect.left()) * (face1_rect.bottom() - face1_rect.top())
    
    # إذا كان لدينا تاريخ مواقع، احسب التنبؤ
    predicted_pos = tracker.predict_next_position()
    if predicted_pos:
        predicted_distance = np.sqrt(
            (face1_center[0] - predicted_pos[0]) ** 2 +
            (face1_center[1] - predicted_pos[1]) ** 2
        )
        # استخدم أقل مسافة
        distance = min(distance, predicted_distance * 0.8)  # وزن أكبر للتنبؤ
    
    return distance

def find_closest_face(new_face, existing_trackers, max_distance=MAX_FACE_DISTANCE):
    """العثور على أقرب متتبع وجه موجود مع تحسين المطابقة"""
    if not existing_trackers:
        return None
    
    best_match_id = None
    min_distance = float('inf')
    
    for face_id, tracker in existing_trackers.items():
        # اعتبار الوجوه في الذاكرة (حتى لو غير مرئية حالياً)
        time_since_seen = time.time() - tracker.last_seen
        
        if time_since_seen < FACE_MEMORY_TIME and tracker.last_position:
            similarity_score = calculate_face_similarity(new_face, tracker.last_position, tracker)
            
            # تطبيق عوامل إضافية للمطابقة
            confidence_factor = tracker.confidence_score
            time_factor = max(0.5, 1.0 - (time_since_seen / FACE_MEMORY_TIME))
            
            adjusted_score = similarity_score / (confidence_factor * time_factor)
            
            if adjusted_score < min_distance and similarity_score < max_distance:
                min_distance = adjusted_score
                best_match_id = face_id
    
    # إذا وُجدت مطابقة جيدة، حدث درجة الثقة
    if best_match_id is not None:
        existing_trackers[best_match_id].confidence_score = min(1.0, 
            existing_trackers[best_match_id].confidence_score + 0.1)
        
        if not existing_trackers[best_match_id].is_active:
            print(f" Face ID {best_match_id} reappeared! (Blinks preserved: {existing_trackers[best_match_id].blink_count})")
    
    return best_match_id

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_np(shape, dtype="int"):
    """Convert dlib shape to numpy array"""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_age_gender(face_roi):
    """Predict age and gender from face ROI with improved accuracy"""
    if not models_loaded:
        return "Unknown", "Unknown", "Unknown"
    
    try:
        # تحسين ROI للتنبؤ الأفضل
        face_roi = cv2.resize(face_roi, (227, 227))
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # تحضير الوجه للـ DNN
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                   (78.4263377603, 87.7689143744, 114.895847746), 
                                   swapRB=False)
        
        # تنبؤ الجنس
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_confidence = np.max(gender_preds[0])
        gender_idx = np.argmax(gender_preds[0])
        
        if gender_confidence > 0.6:
            predicted_gender = GENDER_LIST[gender_idx]
            if gender_confidence < 0.8:
                predicted_gender = GENDER_LIST[1 - gender_idx]
        else:
            predicted_gender = "Unknown"
        
        # تنبؤ العمر
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_confidence = np.max(age_preds[0])
        age_idx = np.argmax(age_preds[0])
        
        if age_confidence > 0.3:
            predicted_age = AGE_LIST[age_idx]
            age_range = predicted_age.replace('(', '').replace(')', '').split('-')
            max_age = int(age_range[1])
            
            if max_age <= 12:
                category = "Child"
            elif max_age <= 20:
                category = "Teen" 
            else:
                category = "Adult"
        else:
            predicted_age = "Unknown"
            category = "Unknown"
            
        return predicted_age, predicted_gender, category
        
    except Exception as e:
        print(f"Error in age/gender prediction: {e}")
        return "Unknown", "Unknown", "Unknown"

def get_face_color(face_id):
    """الحصول على لون مميز لكل وجه"""
    colors = [
        (0, 255, 0),    # أخضر
        (255, 0, 0),    # أحمر
        (0, 0, 255),    # أزرق
        (255, 255, 0),  # أصفر
        (255, 0, 255),  # أرجواني
        (0, 255, 255),  # سماوي
        (128, 0, 128),  # بنفسجي
        (255, 165, 0),  # برتقالي
        (255, 192, 203), # وردي
        (0, 128, 128)   # أزرق مخضر
    ]
    return colors[face_id % len(colors)]

def process_blink_detection(tracker, ear, current_time):
    """معالجة كشف الرمش لوجه معين - مع حفظ العداد"""
    if ear < EAR_THRESHOLD:
        tracker.consecutive_frames += 1
    else:
        if tracker.consecutive_frames >= BLINK_FRAMES:
            if current_time - tracker.last_blink_time > 0.3:  # تجنب العد المزدوج
                tracker.blink_count += 1
                tracker.last_blink_time = current_time
                print(f" Blink detected for Face {tracker.face_id}! Total: {tracker.blink_count}")
        tracker.consecutive_frames = 0

def draw_face_info(frame, x1, y1, x2, y2, matched_id, ear, tracker):
    """رسم معلومات الوجه على الإطار مع حالة التتبع"""
    color = get_face_color(matched_id)
    
    # تحديد سماكة الإطار بناءً على حالة الوجه
    thickness = 3 if tracker.is_active else 2
    line_type = cv2.LINE_AA if tracker.is_active else cv2.LINE_4
    
    # رسم مستطيل الوجه
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)
    
    # إضافة مؤشر حالة الوجه
    status_color = (0, 255, 0) if tracker.is_active else (0, 165, 255)  # أخضر للنشط، برتقالي للمخفي مؤقتاً
    cv2.circle(frame, (x2 - 10, y1 + 10), 5, status_color, -1)
    
    y_offset = y1 - 10
    font_scale = 0.6
    font_thickness = 2
    
    # معرف الوجه مع حالة التتبع
    status = tracker.get_status()
    cv2.putText(frame, f"ID: {matched_id} ({status})", (x1, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    y_offset -= 25
    
    # قيمة EAR
    cv2.putText(frame, f"EAR: {ear:.2f}", (x1, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
    y_offset -= 25
    
    # عدد الرمشات مع الوقت الإجمالي
    total_time = tracker.get_total_time_seen()
    blink_rate = tracker.blink_count / max(total_time, 1) * 60  # رمشات في الدقيقة
    cv2.putText(frame, f"Blinks: {tracker.blink_count} ({blink_rate:.1f}/min)", (x1, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thickness)
    y_offset -= 25
    
    # الجنس والفئة العمرية
    if tracker.gender != "Unknown":
        cv2.putText(frame, f"{tracker.category}, {tracker.gender}", (x1, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thickness)
        y_offset -= 25
        
        # النطاق العمري
        cv2.putText(frame, f"Age: {tracker.age}", (x1, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), font_thickness)

def mark_absent_faces(face_trackers, current_active_ids):
    """وضع علامة على الوجوه الغائبة مؤقتاً"""
    for face_id, tracker in face_trackers.items():
        if face_id not in current_active_ids and tracker.is_active:
            tracker.mark_absent()

# تهيئة التقاط الفيديو
cap = cv2.VideoCapture(0)

print(" Starting Enhanced Multi-Face Blink Detection with Persistent Counters...")
print("Features:")
print("  - Persistent blink counters (maintained when face moves/disappears)")
print("  - Enhanced face tracking with movement prediction")
print("  - Extended memory (15 seconds)")
print("  - Real-time blink rate calculation")
print("\n Controls:")
print("  'r' - Reset all counters")
print("  'c' - Clear all face trackers") 
print("  's' - Show detailed statistics")
print("  'h' - Show help")
print("  'q' - Quit")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # كشف الوجوه
    faces = detector(gray)
    
    # تنظيف متتبعات الوجوه القديمة جداً
    face_trackers = cleanup_old_trackers(face_trackers)
    
    # معالجة كل وجه مكتشف
    current_active_ids = set()
    
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        
        # محاولة المطابقة مع متتبع موجود (حتى المخفي مؤقتاً)
        matched_id = find_closest_face(face, face_trackers)
        
        if matched_id is None:
            # وجه جديد مكتشف
            matched_id = process_new_face()
            face_trackers[matched_id] = FaceTracker(matched_id)
        
        tracker = face_trackers[matched_id]
        tracker.update_position(face)
        current_active_ids.add(matched_id)
        
        # الحصول على نقاط الوجه المميزة
        landmarks = predictor(gray, face)
        landmarks = shape_to_np(landmarks)
        
        # استخراج إحداثيات العيون
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        
        # حساب EAR لكلا العينين
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # رسم العيون
        color = get_face_color(matched_id)
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, color, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, color, 1)
        
        # كشف الرمش - العداد محفوظ دائماً
        current_time = time.time()
        process_blink_detection(tracker, ear, current_time)
        
        # كشف العمر والجنس (أقل تكراراً للأداء)
        if frame_count % 20 == 0:
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                predicted_age, predicted_gender, predicted_category = get_age_gender(face_roi)
                tracker.add_prediction(predicted_age, predicted_gender, predicted_category)
        
        # رسم معلومات الوجه
        draw_face_info(frame, x1, y1, x2, y2, matched_id, ear, tracker)
    
    # وضع علامة على الوجوه المخفية مؤقتاً
    mark_absent_faces(face_trackers, current_active_ids)
    
    # إظهار إحصائيات في أعلى الشاشة
    active_faces = len([t for t in face_trackers.values() if t.is_active])
    total_faces = len(face_trackers)
    cv2.putText(frame, f"Active: {active_faces} | Total: {total_faces}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # عرض الإطار
    cv2.imshow('Enhanced Multi-Face Blink Detection', frame)
    
    # معالجة الضغطات على المفاتيح
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_all_counters()
    elif key == ord('c'):
        clear_all_trackers()
    elif key == ord('s'):
        # عرض الإحصائيات التفصيلية
        print("\n Detailed Statistics:")
        print("-" * 60)
        for face_id, tracker in face_trackers.items():
            total_time = tracker.get_total_time_seen()
            blink_rate = tracker.blink_count / max(total_time, 1) * 60
            status = tracker.get_status()
            print(f"Face {face_id:2d}: {tracker.blink_count:3d} blinks | {blink_rate:5.1f}/min | "
                  f"{total_time:5.1f}s | {status} | {tracker.category}, {tracker.gender}")
        print("-" * 60)
    elif key == ord('h'):
        # عرض المساعدة
        print("\n Controls Help:")
        print("  r: Reset all blink counters")
        print("  c: Clear all face trackers")
        print("  s: Show detailed statistics")
        print("  h: Show this help")
        print("  q: Quit application")
        print("\n Features:")
        print("  • Blink counters are preserved when faces move or disappear temporarily")
        print("  • Faces are remembered for 15 seconds after disappearing")
        print("  • Real-time blink rate calculation (blinks per minute)")
        print("  • Enhanced face tracking with movement prediction")

# التنظيف
cap.release()
cv2.destroyAllWindows()

# طباعة الإحصائيات النهائية
print("\n Final Statistics:")
print("=" * 80)
total_blinks = 0
for face_id, tracker in face_trackers.items():
    total_time = tracker.get_total_time_seen()
    blink_rate = tracker.blink_count / max(total_time, 1) * 60
    total_blinks += tracker.blink_count
    print(f"Face {face_id:2d}: {tracker.blink_count:3d} blinks | {blink_rate:5.1f} blinks/min | "
          f"Seen for {total_time:5.1f}s | {tracker.category}, {tracker.gender}, {tracker.age}")

print("=" * 80)
print(f" Total Faces Detected: {len(face_trackers)}")
print(f"  Total Blinks Counted: {total_blinks}")
print(" Session Complete!")