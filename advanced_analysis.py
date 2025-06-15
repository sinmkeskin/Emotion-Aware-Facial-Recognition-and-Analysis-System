import numpy as np
from datetime import datetime, timedelta
from database import EmotionDatabase

class AdvancedAnalyzer:
    def _init_(self):
        self.db = EmotionDatabase()
        self.emotion_weights = {
            'happy': 1.0,
            'neutral': 0.5,
            'sad': -0.5,
            'angry': -1.0,
            'fear': -0.8,
            'surprise': 0.3,
            'disgust': -0.7
        }

    def calculate_stress_level(self, emotion_history):
        if not emotion_history:
            return 0.0
        
        recent_emotions = emotion_history[:10]  # Son 10 kayıt
        stress_scores = []
        
        for record in recent_emotions:
            try:
                # CSV format değişebileceği için hata kontrolü
                emotion = record[1] if isinstance(record[1], str) and record[1] in self.emotion_weights else "neutral"
                confidence = float(record[2]) if len(record) > 2 and isinstance(record[2], (int, float, str)) else 0.5
                weight = self.emotion_weights.get(emotion, 0)
                stress_scores.append(weight * confidence)
            except (IndexError, ValueError) as e:
                print(f"Kayıt işleme hatası: {e}")
                continue
        
        return np.mean(stress_scores) if stress_scores else 0.0

    def calculate_productivity_score(self, emotion_history):
        if not emotion_history:
            return 0.0
        
        # Son 24 saatteki kayıtları al
        day_ago = datetime.now() - timedelta(days=1)
        recent_emotions = []
        
        for record in emotion_history:
            try:
                # Timestamp yerine doğrudan duygu verisi olabilir
                if len(record) > 1 and isinstance(record[1], str) and record[1] in self.emotion_weights:
                    recent_emotions.append(record)
            except Exception as e:
                print(f"Veri işleme hatası: {e}")
                continue
        
        if not recent_emotions:
            return 0.0
        
        positive_emotions = 0
        for record in recent_emotions:
            try:
                emotion = record[1] if len(record) > 1 else "neutral"
                if emotion in ['happy', 'neutral', 'surprise']:
                    positive_emotions += 1
            except Exception:
                continue
                
        total_emotions = len(recent_emotions)
        
        return (positive_emotions / total_emotions) * 100 if total_emotions > 0 else 0.0

    def estimate_sleep_quality(self, emotion_history):
        if not emotion_history:
            return 0.0
        
        # Gece saatlerindeki duyguları analiz et (22:00 - 06:00)
        night_emotions = []
        current_hour = datetime.now().hour
        
        # Sadece son 24 saat içindeki verileri kullan
        for record in emotion_history[:24]:
            try:
                if len(record) > 1 and isinstance(record[1], str) and record[1] in self.emotion_weights:
                    night_emotions.append(record)
            except Exception:
                continue
        
        if not night_emotions:
            return 0.0
        
        # Gece duygularının ortalamasını hesapla
        night_scores = []
        for record in night_emotions:
            try:
                emotion = record[1] if len(record) > 1 else "neutral"
                confidence = float(record[2]) if len(record) > 2 else 0.5
                night_scores.append(self.emotion_weights.get(emotion, 0) * confidence)
            except (IndexError, ValueError, TypeError):
                continue
                
        return np.mean(night_scores) if night_scores else 0.0

    def analyze_emotion_trends(self, emotion_history, days=7):
        if not emotion_history:
            return {}
        
        # Duygu dağılımını hesapla
        emotion_counts = {}
        for record in emotion_history:
            try:
                if len(record) > 1 and isinstance(record[1], str):
                    emotion = record[1]
                    if emotion in self.emotion_weights:
                        if emotion not in emotion_counts:
                            emotion_counts[emotion] = 0
                        emotion_counts[emotion] += 1
            except Exception:
                continue
        
        # Günlere göre duygu dağılımı yerine genel dağılım döndür
        return {"all_days": emotion_counts}

    def get_group_emotion_analysis(self, faces_data):
        if not faces_data:
            return "neutral", {}
        
        emotions = [face['emotion'] for face in faces_data]
        confidences = [face['confidence'] for face in faces_data]
        
        # En yüksek güvenilirlikli duyguyu grup duygusu olarak al
        max_conf_idx = np.argmax(confidences)
        group_emotion = emotions[max_conf_idx]
        
        # Duygu dağılımını hesapla
        emotion_distribution = {}
        for emotion, conf in zip(emotions, confidences):
            if emotion not in emotion_distribution:
                emotion_distribution[emotion] = 0
            emotion_distribution[emotion] += conf
        
        return group_emotion, emotion_distribution

    def save_analysis(self):
        try:
            emotion_history = self.db.get_emotion_history()
            
            # Boş veri kontrolü
            if not emotion_history or len(emotion_history) == 0:
                return {
                    'stress_level': 0.0,
                    'productivity_score': 0.0,
                    'sleep_quality': 0.0,
                    'analysis_data': {
                        'emotion_trends': {},
                        'timestamp': datetime.now().isoformat()
                    }
                }
            
            # Duygu verilerini düzgün bir şekilde ayıkla
            clean_history = []
            for record in emotion_history:
                try:
                    # Her kayıt en az 3 alana sahip olmalı ve 2. alan bir duygu olmalı
                    if len(record) >= 3 and isinstance(record[1], str) and record[1] in self.emotion_weights:
                        clean_history.append(record)
                except Exception as e:
                    print(f"Kayıt işlenirken hata: {e}")
                    continue
            
            # Eğer temiz veri yoksa, boş sonuç döndür
            if not clean_history:
                return {
                    'stress_level': 0.0,
                    'productivity_score': 0.0,
                    'sleep_quality': 0.0,
                    'analysis_data': {
                        'emotion_trends': {},
                        'timestamp': datetime.now().isoformat()
                    }
                }
            
            stress_level = self.calculate_stress_level(clean_history)
            productivity_score = self.calculate_productivity_score(clean_history)
            sleep_quality = self.estimate_sleep_quality(clean_history)
            
            # Duygu trendlerini hesapla
            emotion_counts = {}
            for record in clean_history:
                try:
                    emotion = record[1]
                    if emotion not in emotion_counts:
                        emotion_counts[emotion] = 0
                    emotion_counts[emotion] += 1
                except Exception:
                    continue
            
            analysis_data = {
                'emotion_trends': {"all_days": emotion_counts},
                'timestamp': datetime.now().isoformat()
            }
            
            try:
                self.db.add_advanced_analysis(
                    stress_level,
                    productivity_score,
                    sleep_quality,
                    analysis_data
                )
            except Exception as e:
                print(f"Veritabanına kayıt hatası: {e}")
            
            return {
                'stress_level': stress_level,
                'productivity_score': productivity_score,
                'sleep_quality': sleep_quality,
                'analysis_data': analysis_data
            }
        except Exception as e:
            print(f"Analiz yaparken genel hata: {e}")
            # Hata durumunda da bir sonuç döndür
            return {
                'stress_level': 0.0,
                'productivity_score': 0.0,
                'sleep_quality': 0.0,
                'analysis_data': {
                    'emotion_trends': {"all_days": {}},
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
            }

    def close(self):
        self.db.close()