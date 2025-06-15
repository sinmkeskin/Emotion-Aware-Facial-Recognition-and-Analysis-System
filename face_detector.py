# face_detector.py
import cv2
import os
import numpy as np
import face_recognition
from database import EmotionDatabase
from emotion_model import EmotionRecognizer


def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces


def detect_multiple_faces(frame, emotion_recognizer=None):
    faces = detect_face(frame)
    faces_data = []
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        # Yüz tanıma
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face, num_jitters=3)
        
        # Duygu analizi
        emotion = "unknown"
        confidence = 0.0
        if emotion_recognizer:
            try:
                emotion, confidence = emotion_recognizer.predict_emotion(face_img)
            except Exception as e:
                print(f"Duygu analizi hatası: {e}")
        
        face_data = {
            'location': (x, y, w, h),
            'encoding': encodings[0] if encodings else None,
            'emotion': emotion,
            'confidence': confidence
        }
        faces_data.append(face_data)
    
    return faces_data


def load_known_faces(folder='known_faces'):
    known_encodings = []
    known_names = []

    if not os.path.exists(folder):
        print(f"Uyarı: {folder} klasörü bulunamadı.")
        # Klasörü oluştur
        os.makedirs(folder, exist_ok=True)
        return known_encodings, known_names

    files = [f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    if not files:
        print("Uyarı: Yüz veritabanında kayıtlı yüz bulunamadı.")
        return known_encodings, known_names
        
    print(f"Yüz veritabanında {len(files)} dosya bulundu, yükleniyor...")

    for file in files:
        if not (file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')):
            continue
            
        try:
            img_path = os.path.join(folder, file)
            img = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(img, num_jitters=5)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(os.path.splitext(file)[0])
                print(f"Yüz yüklendi: {os.path.splitext(file)[0]}")
            else:
                print(f"Uyarı: {file} dosyasında yüz tespit edilemedi.")
        except Exception as e:
            print(f"Hata: {file} dosyası yüklenirken hata oluştu: {e}")

    print(f"Toplam {len(known_names)} yüz yüklendi: {', '.join(known_names)}")
    return known_encodings, known_names


def recognize_faces_in_frame(frame, known_encodings, known_names, emotion_recognizer=None):
    # Frame'de tüm yüzleri tespit et
    faces = detect_face(frame)
    recognized_faces = []
    
    # Hiç kayıtlı yüz yoksa
    if not known_encodings or not known_names:
        print("Uyarı: Yüz veritabanında kayıtlı yüz bulunmadığı için yüz tanıma yapılamıyor.")
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        # Yüz tanıma
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face, num_jitters=3)
        
        # İsim belirleme
        name = "Unknown"
        face_distance = 1.0  # En uzak mesafe
        
        if encodings and len(known_encodings) > 0:
            # En iyi eşleşmeyi bul
            current_encoding = encodings[0]
            
            # Tüm bilinen yüzlerle mesafeleri hesapla
            face_distances = face_recognition.face_distance(known_encodings, current_encoding)
            
            # En yakın eşleşme
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # Eşleşme toleransı (daha düşük -> daha kesin eşleşme)
                tolerance = 0.6  # Daha yüksek değer = daha esnek eşleşme
                
                if best_match_distance <= tolerance:
                    name = known_names[best_match_index]
                    face_distance = best_match_distance
                    print(f"Yüz tanındı: {name} (mesafe: {face_distance:.2f})")
                else:
                    print(f"Uyarı: En yakın yüz {known_names[best_match_index]} ancak mesafe yüksek: {best_match_distance:.2f} > {tolerance}")
        
        # Duygu analizi
        emotion = "unknown"
        confidence = 0.0
        if emotion_recognizer:
            try:
                emotion, confidence = emotion_recognizer.predict_emotion(face_img)
            except Exception as e:
                print(f"Duygu analizi hatası: {e}")
        
        face_data = {
            'location': (x, y, w, h),
            'name': name,
            'face_distance': face_distance,
            'emotion': emotion,
            'confidence': confidence
        }
        recognized_faces.append(face_data)
    
    return recognized_faces


def draw_faces_on_frame(frame, faces_data):
    for face in faces_data:
        x, y, w, h = face['location']
        name = face.get('name', 'Unknown')
        emotion = face.get('emotion', 'unknown')
        confidence = face.get('confidence', 0.0)
        face_distance = face.get('face_distance', 1.0)
        
        # Türkçe duygu isimleri
        emotion_tr = {
            'happy': 'Mutlu', 
            'sad': 'Uzgunn', 
            'angry': 'Kizgin',
            'fear': 'Korku', 
            'surprise': 'Saskin', 
            'disgust': 'Igrenme', 
            'neutral': 'Notr'
        }.get(emotion, emotion)
        
        # Çerçeve rengi (tanınıyorsa yeşil, değilse kırmızı)
        if name != "Unknown":
            # Tanınma güvenine göre renk (daha düşük mesafe = daha iyi eşleşme)
            if face_distance < 0.4:  # Çok iyi eşleşme
                frame_color = (0, 255, 0)  # Koyu yeşil
            elif face_distance < 0.55:  # İyi eşleşme
                frame_color = (0, 255, 128)  # Açık yeşil
            else:  # Sınırda eşleşme
                frame_color = (0, 165, 255)  # Turuncu
        else:
            frame_color = (0, 0, 255)  # Kırmızı
        
        # Yüz çerçevesi
        cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
        
        # İsim ve duygu etiketi
        label = f"{name} | {emotion_tr} ({confidence:.2f})"
        y_pos = max(y - 10, 10)  # Çerçevenin üstünde kalmak için
        
        # Metin arka planı
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y_pos - text_size[1] - 5), (x + text_size[0] + 5, y_pos + 5), frame_color, -1)
        
        # Metin yazısı (beyaz)
        cv2.putText(frame, label, (x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame