import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Dropout
import os

# Swish aktivasyon fonksiyonunu tanımla
@register_keras_serializable()
def swish(x):
    return x * tf.sigmoid(x)

# FixedDropout katmanını tanımla
@register_keras_serializable()
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return super().call(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

__all__ = ['EmotionRecognizer']

class EmotionRecognizer:
    def __init__(self, model_path='model/emotion_model.h5'):
        try:
            # Model dosyasının varlığını kontrol et
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

            # TensorFlow ve Keras versiyonlarını kontrol et
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Keras version: {keras.__version__}")
            
            # Model yükleme seçenekleri
            custom_objects = {
                'Adam': tf.keras.optimizers.Adam,
                'InputLayer': tf.keras.layers.InputLayer,
                'Dense': tf.keras.layers.Dense,
                'Conv2D': tf.keras.layers.Conv2D,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'Flatten': tf.keras.layers.Flatten,
                'Dropout': tf.keras.layers.Dropout,
                'FixedDropout': FixedDropout,  # FixedDropout katmanını ekle
                'BatchNormalization': tf.keras.layers.BatchNormalization,
                'swish': swish,  # Swish aktivasyon fonksiyonunu ekle
                'Activation': tf.keras.layers.Activation
            }
            
            # Modeli yükle
            try:
                # tf.keras.models.load_model ile yükle
                self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            except Exception as e:
                print(f"Model yükleme hatası: {str(e)}")
                raise
            
            # Model giriş boyutunu al
            if hasattr(self.model, 'input_shape'):
                self.input_shape = self.model.input_shape[1:3]  # (height, width)
            else:
                # Varsayılan boyut
                self.input_shape = (224, 224)
            
            # Sınıf isimleri
            self.class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            
            print(f"Model başarıyla yüklendi. Giriş boyutu: {self.input_shape}")
            
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {str(e)}")
            raise

    def predict_emotion(self, face_img):
        try:
            # Görüntüyü gri tonlamaya çevir
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Görüntüyü yeniden boyutlandır
            face_img = cv2.resize(face_img, self.input_shape)
            
            # Normalize et
            face_img = face_img.astype('float32') / 255.0
            
            # Model girişi için boyutları ayarla
            if len(self.model.input_shape) == 4:  # (batch, height, width, channels)
                face_img = np.expand_dims(face_img, axis=0)  # batch boyutu ekle
                face_img = np.expand_dims(face_img, axis=-1)  # channel boyutu ekle
            
            # Tahmin yap
            prediction = self.model.predict(face_img, verbose=0)
            emotion_index = np.argmax(prediction)
            
            return self.class_names[emotion_index], prediction[0][emotion_index]
            
        except Exception as e:
            print(f"Duygu tahmini yapılırken hata oluştu: {str(e)}")
            return "unknown", 0.0

# Test için
if __name__ == "__main__":
    try:
        recognizer = EmotionRecognizer()
        print("EmotionRecognizer başarıyla oluşturuldu.")
    except Exception as e:
        print(f"Hata: {str(e)}")

