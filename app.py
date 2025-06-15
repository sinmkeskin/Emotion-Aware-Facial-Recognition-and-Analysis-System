import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
import csv

# Proje dizinini Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from emotion_model import EmotionRecognizer
from face_detector import detect_face, detect_multiple_faces, recognize_faces_in_frame, draw_faces_on_frame, load_known_faces
from emotion_responses import get_response
import face_recognition
import tempfile
from database import EmotionDatabase
from advanced_analysis import AdvancedAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Sayfa yapılandırması
st.set_page_config(
    page_title="Duygu Tanıma Uygulaması",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state ayarları
if 'menu' not in st.session_state:
    st.session_state.menu = "Duygu Analizi"

if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Tema değiştirme fonksiyonu
def toggle_theme():
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Tema ayarları
light_theme = """
<style>
    /* Ana tema renkleri - Daha yumuşak mavi tonları kullanan profesyonel tema */
    .main {
        background-color: #F0F4F8;
        color: #1A365D;
    }
    
    /* Sidebar tema */
    .css-1d391kg {
        background-color: #E2E8F0;
    }
    
    /* Başlıklar */
    h1, h2, h3, h4, h5, h6 {
        color: #2B6CB0 !important;
    }
    
    /* Butonlar */
    .stButton>button {
        background-color: #4299E1;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2B6CB0;
        box-shadow: 0 2px 8px rgba(43, 108, 176, 0.5);
    }
    
    /* Metin kutuları */
    .stTextInput>div>div>input {
        background-color: #EDF2F7;
        color: #1A365D;
        border: 1px solid #A0AEC0;
        border-radius: 5px;
    }
    
    /* Seçim kutuları */
    .stSelectbox>div>div>select {
        background-color: #EDF2F7;
        color: #1A365D;
        border: 1px solid #A0AEC0;
        border-radius: 5px;
    }
    
    /* Başarı mesajları */
    .stSuccess {
        background-color: #F0FFF4;
        border: 1px solid #68D391;
        color: #276749;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Bilgi mesajları */
    .stInfo {
        background-color: #EBF8FF;
        border: 1px solid #63B3ED;
        color: #2C5282;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Uyarı mesajları */
    .stWarning {
        background-color: #FFFBEB;
        border: 1px solid #F6AD55;
        color: #C05621;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Hata mesajları */
    .stError {
        background-color: #FFF5F5;
        border: 1px solid #FC8181;
        color: #C53030;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Grafikler için tema */
    .js-plotly-plot {
        background-color: #FFFFFF !important;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Radio butonlar */
    .stRadio>div {
        background-color: #EDF2F7;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #A0AEC0;
    }
    
    /* Slider */
    .stSlider>div>div>div {
        background-color: #4299E1;
    }
    
    /* Progress bar */
    .stProgress>div>div>div {
        background-color: #4299E1;
    }
    
    /* Kamera görüntüsü container */
    .stCameraInput>div {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #A0AEC0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Dosya yükleme alanı */
    .stFileUploader>div {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #A0AEC0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Kart stilleri */
    .card {
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #4299E1;
    }
    
    /* Tıklanabilir öğelerde hover efekti */
    a:hover, .clickable:hover {
        opacity: 0.8;
        text-decoration: none;
    }
</style>
"""

dark_theme = """
<style>
    /* Ana tema renkleri - Koyu tema */
    .main {
        background-color: #1A202C;
        color: #E2E8F0;
    }
    
    /* Sidebar tema */
    .css-1d391kg {
        background-color: #2D3748;
    }
    
    /* Başlıklar */
    h1, h2, h3, h4, h5, h6 {
        color: #90CDF4 !important;
    }
    
    /* Butonlar */
    .stButton>button {
        background-color: #4299E1;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2B6CB0;
        box-shadow: 0 2px 8px rgba(66, 153, 225, 0.5);
    }
    
    /* Metin kutuları */
    .stTextInput>div>div>input {
        background-color: #2D3748;
        color: #E2E8F0;
        border: 1px solid #4A5568;
        border-radius: 5px;
    }
    
    /* Seçim kutuları */
    .stSelectbox>div>div>select {
        background-color: #2D3748;
        color: #E2E8F0;
        border: 1px solid #4A5568;
        border-radius: 5px;
    }
    
    /* Başarı mesajları */
    .stSuccess {
        background-color: #1C4532;
        border: 1px solid #38A169;
        color: #9AE6B4;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Bilgi mesajları */
    .stInfo {
        background-color: #2A4365;
        border: 1px solid #3182CE;
        color: #90CDF4;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Uyarı mesajları */
    .stWarning {
        background-color: #744210;
        border: 1px solid #D69E2E;
        color: #F6E05E;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Hata mesajları */
    .stError {
        background-color: #742A2A;
        border: 1px solid #E53E3E;
        color: #FC8181;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Grafikler için tema */
    .js-plotly-plot {
        background-color: #2D3748 !important;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Radio butonlar */
    .stRadio>div {
        background-color: #2D3748;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #4A5568;
    }
    
    /* Slider */
    .stSlider>div>div>div {
        background-color: #4299E1;
    }
    
    /* Progress bar */
    .stProgress>div>div>div {
        background-color: #4299E1;
    }
    
    /* Kamera görüntüsü container */
    .stCameraInput>div {
        background-color: #2D3748;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #4A5568;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Dosya yükleme alanı */
    .stFileUploader>div {
        background-color: #2D3748;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #4A5568;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Kart stilleri */
    .card {
        background-color: #2D3748;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #4299E1;
    }
    
    /* Tıklanabilir öğelerde hover efekti */
    a:hover, .clickable:hover {
        opacity: 0.8;
        text-decoration: none;
    }
</style>
"""

# Aktif temayı uygula
st.markdown(light_theme if st.session_state.theme == "light" else dark_theme, unsafe_allow_html=True)

# Başlık ve açıklama
st.title("🤖 Duygu Tanıma Uygulaması")
st.markdown("""
<div class="card">
    Bu uygulama, gerçek zamanlı olarak yüz ifadelerinizi analiz eder ve duygularınızı tespit eder.
    Ayrıca kayıtlı yüzleri tanıyabilir ve size özel yanıtlar verebilir.
</div>
""", unsafe_allow_html=True)

# Sidebar menüsü
st.sidebar.title("Menü")

# Tema değiştirme butonu
theme_label = "🌙 Karanlık Tema" if st.session_state.theme == "light" else "☀️ Aydınlık Tema"
if st.sidebar.button(theme_label):
    toggle_theme()
    st.experimental_rerun()

# Menu seçimi
menu = st.sidebar.radio(
    "İşlem Seçin",
    ["Duygu Analizi", "Yüz Ekle", "Kayıtlı Yüzler", "Duygu Geçmişi"],
    index=["Duygu Analizi", "Yüz Ekle", "Kayıtlı Yüzler", "Duygu Geçmişi"].index(st.session_state.menu)
)

# Session state güncelle
st.session_state.menu = menu

# Model ve tanıyıcıları yükle
@st.cache_resource
def load_models():
    recognizer = EmotionRecognizer()
    known_encodings, known_names = load_known_faces()
    db = EmotionDatabase()
    analyzer = AdvancedAnalyzer()
    return recognizer, known_encodings, known_names, db, analyzer

recognizer, known_encodings, known_names, db, analyzer = load_models()

def normalize_person_name(name):
    if not name or not str(name).strip().strip(','):
        return "Bilinmeyen"
    name = str(name).strip().strip(',').lower()
    if name in ["unknown", "bilinmeyen"]:
        return "Bilinmeyen"
    return name.capitalize()


# Duygu Analizi Sayfası
if menu == "Duygu Analizi":
    st.header("🎥 Gerçek Zamanlı Duygu Analizi")
    
    st.markdown('<div class="card">'
                'Bu bölümde kamera veya yüklenen görüntüden duygu analizi yapabilirsiniz.<br>'
                'Kamera kullanarak gerçek zamanlı analiz yapabilir veya bir fotoğraf yükleyebilirsiniz.'
                '</div>', unsafe_allow_html=True)
    
    # Kamera seçimi
    camera_option = st.radio("Kamera Seçin:", ["Webcam", "Dosyadan Yükle"])
    
    if camera_option == "Webcam":
        # Webcam görüntüsü
        st.markdown('<div style="margin-top: 20px; margin-bottom: 10px;">'
                    '<h4>📸 Kameraya bakarak duygu analizini başlatın</h4>'
                    '</div>', unsafe_allow_html=True)
        
        img_file_buffer = st.camera_input("Kamera")
        
        if img_file_buffer is not None:
            # Görüntüyü numpy dizisine dönüştür
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("Yüz tanıma ve duygu analizi yapılıyor..."):
                # Çoklu yüz analizi - doğrudan tanıma fonksiyonunu kullan
                recognized_faces = recognize_faces_in_frame(cv2_img, known_encodings, known_names, recognizer)
                
                if recognized_faces:
                    # Görüntüyü işaretle
                    cv2_img = draw_faces_on_frame(cv2_img, recognized_faces)
                    
                    # Sonuçları göster
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.success(f"👥 Tespit Edilen Yüz Sayısı: {len(recognized_faces)}")
                        for face in recognized_faces:
                            st.info(f"👤 {face.get('name', 'Bilinmeyen')}: {face.get('emotion', 'bilinmeyen')} ({face.get('confidence', 0.0):.2f})")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Duygu analizini görselleştir
                        if len(recognized_faces) > 0:
                            # Kişiye göre duyguları gruplayalım
                            persons_emotions = {}
                            
                            for face in recognized_faces:
                                name = face.get('name', 'Bilinmeyen')
                                emotion = face.get('emotion', 'unknown')
                                confidence = face.get('confidence', 0.0)
                                
                                if name not in persons_emotions:
                                    persons_emotions[name] = []
                                
                                persons_emotions[name].append({
                                    'emotion': emotion,
                                    'confidence': confidence
                                })
                            
                            # Kişi sayısı duygu analiz grafiği için
                            all_emotions = []
                            all_confidences = []
                            all_names = []
                            
                            for name, emotions in persons_emotions.items():
                                for e in emotions:
                                    all_emotions.append(e['emotion'])
                                    all_confidences.append(e['confidence'])
                                    all_names.append(name)
                            
                           
                            emotion_labels = {
                                'happy': 'Mutlu',
                                'sad': 'Uzgun',
                                'angry': 'Kızgın',
                                'fear': 'Korku',
                                'surprise': 'Saskın',
                                'disgust': 'Igrenme',
                                'neutral': 'Notr',
                                'unknown': 'Bilinmeyen'
                            }
                            
                            color_map = {
                                'happy': '#4CAF50',    # Yeşil
                                'sad': '#2196F3',      # Mavi
                                'angry': '#F44336',    # Kırmızı
                                'fear': '#9C27B0',     # Mor
                                'surprise': '#FF9800', # Turuncu
                                'disgust': '#795548',  # Kahverengi
                                'neutral': '#607D8B',  # Gri
                                'unknown': '#000000'   # Siyah
                            }
                            
                            # İlk olarak, kişilere göre duygu dağılımını gösteren bir grafik
                            if len(persons_emotions) > 1:
                                st.subheader("Kişilere Göre Duygu Dağılımı")
                                
                                # Veri hazırlama
                                chart_data = []
                                for name, emotions in persons_emotions.items():
                                    for e in emotions:
                                        chart_data.append({
                                            'Kişi': name, 
                                            'Duygu': emotion_labels.get(e['emotion'], e['emotion']),
                                            'Güven': e['confidence']
                                        })
                                
                                # Grafik oluşturma
                                fig = px.bar(
                                    chart_data,
                                    x='Kişi',
                                    y='Güven',
                                    color='Duygu',
                                    labels={'Güven': 'Güven Skoru', 'Kişi': 'Kişi Adı'},
                                    title="Kişilere Göre Duygu Analizi"
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font={'color': '#1A365D'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Genel duygu dağılımı grafiği
                            fig = px.bar(
                                x=[emotion_labels.get(e, e) for e in all_emotions],
                                y=all_confidences,
                                labels={'x': 'Duygu', 'y': 'Güven Skoru'},
                                title="Genel Duygu Analizi Sonuçları",
                                color=all_emotions,
                                color_discrete_map=color_map,
                                hover_data={"Kişi": all_names}
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font={'color': '#1A365D'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        # İlk yüz için öneri al
                        if recognized_faces:
                            emotion = recognized_faces[0].get('emotion', 'neutral')
                            response = get_response(emotion)
                            st.markdown(f"<h3>💭 Öneriler:</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p style='font-size: 1.1em; padding: 10px;'>{response}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Görüntüyü göster
                    st.image(cv2_img, channels="BGR", use_column_width=True)
                    
                    # Veritabanına kaydet
                    for face in recognized_faces:
                        db.add_emotion_record(face.get('emotion', 'unknown'), face.get('confidence', 0.0), face.get('name', 'Bilinmeyen'))
                    
                    # Analiz Sonrası Butonlar
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("<h4>Analiz Sonrası İşlemler</h4>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📊 Duygu Geçmişine Git"):
                            st.session_state.menu = "Duygu Geçmişi"
                            st.experimental_rerun()
                    
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Yüz tespit edilemedi. Lütfen kameraya düzgün bakın.")
                    st.markdown("<div class='card'>"
                                "<h4>İpuçları:</h4>"
                                "<ul>"
                                "<li>Yüzünüzün kamera açısında olduğundan emin olun</li>"
                                "<li>Yeterli ışık olduğundan emin olun</li>"
                                "<li>Kameraya direkt bakın</li>"
                                "<li>Engelleri kaldırın (gözlük, şapka vb.)</li>"
                                "</ul>"
                                "</div>", unsafe_allow_html=True)
    
    else:  # Dosyadan Yükle
        st.markdown('<div style="margin-top: 20px; margin-bottom: 10px;">'
                    '<h4>📁 Bir görüntü yükleyin</h4>'
                    '</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Bir görüntü seçin", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Dosyayı göster
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görüntü", use_column_width=True)
            
            # İşleme buton
            if st.button("🔍 Duygu Analizi Yap"):
                with st.spinner("Duygu analizi yapılıyor..."):
                    cv2_img = np.array(image)
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
                    
                    # Çoklu yüz analizi - doğrudan tanıma fonksiyonunu kullan
                    recognized_faces = recognize_faces_in_frame(cv2_img, known_encodings, known_names, recognizer)
                    
                    if recognized_faces:
                        # Görüntüyü işaretle
                        cv2_img = draw_faces_on_frame(cv2_img, recognized_faces)
                        
                        st.subheader("📊 Analiz Sonuçları")
                        
                        # Sonuçları göster
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.success(f"👥 Tespit Edilen Yüz Sayısı: {len(recognized_faces)}")
                            for face in recognized_faces:
                                st.info(f"👤 {face.get('name', 'Bilinmeyen')}: {face.get('emotion', 'bilinmeyen')} ({face.get('confidence', 0.0):.2f})")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Duygu analizini görselleştir
                            if len(recognized_faces) > 0:
                                # Kişiye göre duyguları gruplayalım
                                persons_emotions = {}
                                
                                for face in recognized_faces:
                                    name = face.get('name', 'Bilinmeyen')
                                    emotion = face.get('emotion', 'unknown')
                                    confidence = face.get('confidence', 0.0)
                                    
                                    if name not in persons_emotions:
                                        persons_emotions[name] = []
                                    
                                    persons_emotions[name].append({
                                        'emotion': emotion,
                                        'confidence': confidence
                                    })
                                
                                # Kişi sayısı duygu analiz grafiği için
                                all_emotions = []
                                all_confidences = []
                                all_names = []
                                
                                for name, emotions in persons_emotions.items():
                                    for e in emotions:
                                        all_emotions.append(e['emotion'])
                                        all_confidences.append(e['confidence'])
                                        all_names.append(name)
                                
                                # Türkçe duygu isimleri
                                emotion_labels = {
                                    'happy': 'Mutlu',
                                    'sad': 'Üzgün',
                                    'angry': 'Kızgın',
                                    'fear': 'Korku',
                                    'surprise': 'Saskın',
                                    'disgust': 'Igrenme',
                                    'neutral': 'Notr',
                                    'unknown': 'Bilinmeyen'
                                }
                                
                                # Renk eşleştirmeleri
                                color_map = {
                                    'happy': '#4CAF50',    # Yeşil
                                    'sad': '#2196F3',      # Mavi
                                    'angry': '#F44336',    # Kırmızı
                                    'fear': '#9C27B0',     # Mor
                                    'surprise': '#FF9800', # Turuncu
                                    'disgust': '#795548',  # Kahverengi
                                    'neutral': '#607D8B',  # Gri
                                    'unknown': '#000000'   # Siyah
                                }
                                
                                # İlk olarak, kişilere göre duygu dağılımını gösteren bir grafik
                                if len(persons_emotions) > 1:
                                    st.subheader("Kişilere Göre Duygu Dağılımı")
                                    
                                    # Veri hazırlama
                                    chart_data = []
                                    for name, emotions in persons_emotions.items():
                                        for e in emotions:
                                            chart_data.append({
                                                'Kişi': name, 
                                                'Duygu': emotion_labels.get(e['emotion'], e['emotion']),
                                                'Güven': e['confidence']
                                            })
                                    
                                    # Grafik oluşturma
                                    fig = px.bar(
                                        chart_data,
                                        x='Kişi',
                                        y='Güven',
                                        color='Duygu',
                                        labels={'Güven': 'Güven Skoru', 'Kişi': 'Kişi Adı'},
                                        title="Kişilere Göre Duygu Analizi"
                                    )
                                    fig.update_layout(
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        font={'color': '#1A365D'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Genel duygu dağılımı grafiği
                                fig = px.bar(
                                    x=[emotion_labels.get(e, e) for e in all_emotions],
                                    y=all_confidences,
                                    labels={'x': 'Duygu', 'y': 'Güven Skoru'},
                                    title="Genel Duygu Analizi Sonuçları",
                                    color=all_emotions,
                                    color_discrete_map=color_map,
                                    hover_data={"Kişi": all_names}
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font={'color': '#1A365D'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            if recognized_faces:
                                emotion = recognized_faces[0].get('emotion', 'neutral')
                                response = get_response(emotion)
                                st.markdown(f"<h3>💭 Öneriler:</h3>", unsafe_allow_html=True)
                                st.markdown(f"<p style='font-size: 1.1em; padding: 10px;'>{response}</p>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # İşaretli görüntüyü göster
                        st.subheader("🖼 İşaretli Görüntü")
                        st.image(cv2_img, channels="BGR", use_column_width=True)
                        
                        # Veritabanına kaydet
                        for face in recognized_faces:
                            db.add_emotion_record(face.get('emotion', 'unknown'), face.get('confidence', 0.0), face.get('name', 'Bilinmeyen'))
                        
                        # Analiz Sonrası Butonlar
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("<h4>Analiz Sonrası İşlemler</h4>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📊 Duygu Geçmişine Git"):
                                st.session_state.menu = "Duygu Geçmişi"
                                st.experimental_rerun()
                        
                        with col2:
                            if st.button("🔄 Yeni Görüntü Yükle"):
                                st.experimental_rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Yüz tespit edilemedi. Lütfen başka bir görüntü deneyin.")
                        st.markdown("<div class='card'>"
                                    "<h4>İpuçları:</h4>"
                                    "<ul>"
                                    "<li>Görüntüde net bir yüz olduğundan emin olun</li>"
                                    "<li>Görüntünün kalitesi yeterli olmalıdır</li>"
                                    "<li>Farklı bir açıdan çekilmiş görüntü deneyin</li>"
                                    "<li>İyi aydınlatılmış bir görüntü kullanın</li>"
                                    "</ul>"
                                    "</div>", unsafe_allow_html=True)

# Duygu Geçmişi Sayfası
elif menu == "Duygu Geçmişi":
    st.header("📊 Duygu Geçmişi")
    
    # Duygu geçmişini DataFrame olarak al
    try:
        df = pd.read_csv("data/emotion_history.csv")
    except Exception as e:
        st.error(f"CSV okunamadı: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Kişi adlarını normalize et
        def normalize_person_name(name):
            if pd.isna(name) or not str(name).strip().strip(','):
                return "Bilinmeyen"
            name = str(name).strip().strip(',').lower()
            if name in ["unknown", "bilinmeyen"]:
                return "Bilinmeyen"
            return name.capitalize()
        df["face_id"] = df["face_id"].apply(normalize_person_name)

        persons = ["Tümü"] + sorted(df["face_id"].unique())
        selected_person = st.selectbox("Kişi Seçin:", persons)

        if selected_person == "Tümü":
            filtered_df = df
        else:
            filtered_df = df[df["face_id"] == selected_person]

        if not filtered_df.empty:
            st.subheader(f"{selected_person} Duygu Geçmişi Özeti")

            # Duygu dağılımı
            emotion_counts = filtered_df["emotion"].value_counts()
            total = emotion_counts.sum()

            # 1. Kartlar/rozetler
            st.markdown("### Duygu Sayıları")
            cols = st.columns(len(emotion_counts))
            emoji_map = {
                "happy": "😊",
                "neutral": "😐",
                "sad": "😢",
                "angry": "😡",
                "surprise": "😲",
                "disgust": "🤢",
                "fear": "😱",
                "unknown": "❓"
            }
            for i, (emotion, count) in enumerate(emotion_counts.items()):
                emoji = emoji_map.get(emotion, "❓")
                cols[i].markdown(f"<div style='text-align:center; font-size:2em'>{emoji}</div>", unsafe_allow_html=True)
                cols[i].metric(emotion.capitalize(), count, f"%{(count/total*100):.1f}")

            # 2. Pasta grafik
            st.markdown("### Duygu Dağılımı Grafiği")
            fig = px.pie(
                values=emotion_counts.values,
                names=[emoji_map.get(e, e) + ' ' + e.capitalize() for e in emotion_counts.index],
                title="Duygu Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 3. Kısa özet cümle
            summary = ", ".join([f"{emoji_map.get(e, e)} {e.capitalize()}: {count} (%{(count/total*100):.1f})"
                                 for e, count in emotion_counts.items()])
            st.info(f"Toplam {total} kayıt: {summary}")
        else:
            st.info("Henüz kayıtlı duygu verisi bulunmuyor.")
    else:
        st.info("Henüz kayıtlı duygu verisi bulunmuyor.")

# Yüz Ekleme Sayfası
elif menu == "Yüz Ekle":
    st.header("➕ Yeni Yüz Ekle")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('''
    <h4>Yüz Tanıma Veritabanına Kişi Ekleyin</h4>
    <p>Duygu analizi yaparken tanınan kişiler için isimle birlikte duygu analizi yapılacaktır. Yüz ekleme adımları:</p>
    <ol>
        <li>Kişinin adını girin</li>
        <li>Kamera ile net bir fotoğraf çekin</li>
        <li>Kişinin yüzü doğru tespit edildiğinde "Yüzü Kaydet" butonuna tıklayın</li>
    </ol>
    <p><strong>Not:</strong> Yüz veritabanına eklenen kişiler duygu analizi yapıldığında otomatik olarak tanınacaktır.</p>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    name = st.text_input("Kişinin Adı:")
    
    # Ön kontrol
    if name.strip() == "":
        st.warning("Lütfen kaydetmek istediğiniz kişinin adını girin.")
    else:
        # Fotoğraf çekme
        img_file_buffer = st.camera_input("Kamera ile fotoğraf çekin")
        
        if img_file_buffer is not None:
            # Görüntüyü numpy dizisine dönüştür
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Yüz tespiti yap
            faces = detect_face(cv2_img)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = cv2_img[y:y+h, x:x+w]
                
                # Yüzü göster
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Tespit Edilen Yüz")
                
                # Görüntüyü işaretle
                cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(cv2_img, channels="BGR", caption="Tespit Edilen Yüz", use_column_width=True)
                
                # Kırpılmış yüz
                st.image(face_img, channels="BGR", caption="Kırpılmış Yüz", width=150)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Kaydet butonu
                if st.button("💾 Yüzü Kaydet"):
                    # Klasörü kontrol et
                    save_path = 'known_faces'
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Dosya adını oluştur
                    img_path = os.path.join(save_path, f"{name}.jpg")
                    
                    # Eğer dosya zaten varsa
                    if os.path.exists(img_path):
                        overwrite = st.checkbox(f"{name} isimli kayıt zaten var. Üzerine yazmak istiyor musunuz?")
                        if overwrite:
                            cv2.imwrite(img_path, face_img)
                            st.success(f"✅ {name} isimli yüz güncellendi!")
                            # Yüz veritabanını yeniden yükle
                            st.session_state.reload_faces = True
                    else:
                        # Yüzü kaydet
                        cv2.imwrite(img_path, face_img)
                        st.success(f"✅ {name} isimli yüz başarıyla kaydedildi!")
                        # Yüz veritabanını yeniden yükle
                        st.session_state.reload_faces = True
                        
                    # Kaydedilen yüzleri göster
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Kayıtlı Yüzler")
                    
                    # Tüm kayıtlı yüzleri listeleme
                    files = [f for f in os.listdir(save_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
                    if not files:
                        st.info("Kayıtlı kişi bulunamadı.")
                    else:
                        st.success(f"Toplam {len(files)} kişi kayıtlı.")
                        # 3 sütunlu düzende göster
                        cols = st.columns(3)
                        for idx, file in enumerate(files):
                            person_name = os.path.splitext(file)[0]
                            img_path = os.path.join(save_path, file)
                            img = Image.open(img_path)
                            
                            with cols[idx % 3]:
                                st.image(img, caption=person_name, width=150)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Yüz tespit edilemedi. Lütfen kameraya düzgün bakın ve tekrar deneyin.")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('''
                <h4>Yüz Tespiti İpuçları:</h4>
                <ul>
                    <li>İyi aydınlatılmış bir ortamda olun</li>
                    <li>Kameraya direkt bakın</li>
                    <li>Yüzünüzü tam olarak gösterin</li>
                    <li>Gözlük, şapka gibi aksesuarları çıkarın</li>
                </ul>
                ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Kayıtlı Yüzler Sayfası
elif menu == "Kayıtlı Yüzler":
    st.header("👥 Kayıtlı Yüzler")
    
    folder = 'known_faces'
    if not os.path.exists(folder):
        st.warning("Kayıtlı kişi klasörü bulunamadı.")
    else:
        files = [f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]
        if not files:
            st.info("Kayıtlı kişi bulunmuyor.")
        else:
            cols = st.columns(3)
            for idx, file in enumerate(files):
                name = os.path.splitext(file)[0]
                img_path = os.path.join(folder, file)
                img = Image.open(img_path)
                
                with cols[idx % 3]:
                    st.image(img, caption=name, use_column_width=True)
                    if st.button(f"Sil {name}", key=name):
                        os.remove(img_path)
                        st.success(f"{name} silindi!")
                        st.rerun()