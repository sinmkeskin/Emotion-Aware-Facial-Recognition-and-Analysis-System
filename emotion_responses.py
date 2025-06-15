import os
import requests

# Sabit API anahtarı (güvenlik açısından normalde ortam değişkeni olarak saklanmalıdır)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "<YOUR_APİ_KEY>")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"  # veya başka bir model

# Eğer API çağrısı yapılamazsa kullanılacak sabit yanıtlar
DEFAULT_RESPONSES = {
    "happy": "Mutlu olduğunu görmek harika! Bu enerjiyi sürdürmek için sevdiğin bir şarkı dinleyebilir veya arkadaşlarınla vakit geçirebilirsin. Bugün, mutluluğunu paylaşarak etrafındakilere de pozitif enerji verebilirsin.",
    "sad": "Üzgün hissetmek de normal. Kendine biraz zaman ayırabilir, sevdiğin bir filmi izleyebilir veya rahatlatıcı bir müzik dinleyebilirsin. Belki sıcak bir çay veya kahve iyi gelebilir. Unutma, her zor duygu geçicidir.",
    "angry": "Öfke, enerjini tüketebilir. Derin nefes alma egzersizleri yapmayı, kısa bir yürüyüşe çıkmayı veya sakinleştirici bir müzik dinlemeyi deneyebilirsin. Duygularını bir yere yazmak da yardımcı olabilir.",
    "surprise": "Şaşkınlık, hayatın bize sunduğu ilginç anlardan biri! Bu duyguyu değerlendir ve yeni keşifler yapmak için bir fırsat olarak gör. Belki de günün geri kalanında başka sürprizler de seni bekliyor olabilir.",
    "fear": "Korku hissettiğinde, güvenli hissettiğin bir yere gitmeyi ve sevdiğin biriyle konuşmayı deneyebilirsin. Derin nefes alıp vermen ve 'şu anda güvendeyim' diye kendine hatırlatman yardımcı olabilir.",
    "disgust": "Hoşnut olmadığın bir durumla karşılaştığında, dikkatini daha pozitif şeylere yönlendirmek iyi gelebilir. Belki sevdiğin bir hobi ile uğraşabilir veya temiz havada kısa bir yürüyüş yapabilirsin.",
    "neutral": "Nötr hissetmek, yeni deneyimlere açık olduğun bir an olabilir. Belki yeni bir kitap okumayı, yeni bir yemek denemeyi veya sevdiğin bir aktiviteye zaman ayırmayı düşünebilirsin."
}

def get_response(emotion):
    if not GROQ_API_KEY:
        # API anahtarı yoksa varsayılan yanıtları kullan
        return DEFAULT_RESPONSES.get(emotion, "Bu duygu için henüz bir önerim yok, ama seninle konuşmak her zaman güzel.")

    try:
        prompt = f"""
Sen bir duygu asistanısın. Kullanıcı şu anda '{emotion}' hissediyor. Ona moral verecek, eğlendirecek veya ilgisini dağıtacak önerilerde bulun. Bunlar arasında bir şarkı önerisi, bir kahve önerisi ve güncel genel kültürden kısa bir bilgi olabilir. Cevabını samimi ve kısa tut. Türkçe yaz.
"""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "Sen yardımcı bir asistanısın."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.8
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        # Hata durumunda varsayılan yanıtları kullan
        print(f"API hatası: {e}")
        return DEFAULT_RESPONSES.get(emotion, "Bu duygu için henüz bir önerim yok, ama seninle konuşmak her zaman güzel.")
