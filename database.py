import csv
from datetime import datetime
import json
import os
import pandas as pd

class EmotionDatabase:
    def __init__(self):
        self.emotion_history_file = 'data/emotion_history.csv'
        self.create_files()

    def create_files(self):
        # Klasör oluştur
        os.makedirs('data', exist_ok=True)
        
        # Duygu geçmişi dosyası
        if not os.path.exists(self.emotion_history_file):
            with open(self.emotion_history_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'emotion', 'confidence', 'face_id'])
        

    def add_emotion_record(self, emotion, confidence, face_id=None, additional_data=None):
        with open(self.emotion_history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            clean_face_id = str(face_id).strip().strip(',') if face_id else ""
            writer.writerow([
                datetime.now().isoformat(),
                emotion,
                confidence,
                clean_face_id
            ])

   
    def get_emotion_history(self, days=7):
        try:
            df = pd.read_csv(self.emotion_history_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            return df.values.tolist()
        except Exception as e:
            print(f"Error reading emotion history: {e}")
            return []


    def get_emotion_stats(self):
        try:
            df = pd.read_csv(self.emotion_history_file)
            if df.empty:
                return {}
            
            # Duygu dağılımı
            emotion_counts = df['emotion'].value_counts().to_dict()
            
            # Ortalama güven skorları
            confidence_means = df.groupby('emotion')['confidence'].mean().to_dict()
            
            return {
                'emotion_counts': emotion_counts,
                'confidence_means': confidence_means
            }
        except Exception as e:
            print(f"Error getting emotion stats: {e}")
            return {}

    def export_to_excel(self, output_file='emotion_analysis.xlsx'):
        try:
            with pd.ExcelWriter(output_file) as writer:
                # Duygu geçmişi
                df_emotion = pd.read_csv(self.emotion_history_file)
                df_emotion.to_excel(writer, sheet_name='Emotion History', index=False)
                
            
            return True
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False