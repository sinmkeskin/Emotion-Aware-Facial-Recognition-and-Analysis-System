# Emotion-Aware-Facial-Recognition-and-Analysis-System

This project is a Python-based system that detects human faces and analyzes their emotions in real-time or from static data. It uses facial recognition and emotion classification models to interpret user expressions and respond accordingly.

## 📁 Project Structure

```
.
├── app.py                  # Main application entry
├── database.py             # Handles database operations
├── emotion_model.py        # Loads and runs emotion classification model
├── face_detector.py        # Face detection logic
├── emotion_responses.py    # Custom responses based on detected emotions
├── advanced_analysis.py    # Additional emotion data processing
├── model/                  # Pre-trained model files
├── data/                   # Additional image/video data
├── known_faces/            # Labeled face images for recognition
├── emotion_history.db      # SQLite database for storing emotion logs
├── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

1. **Clone the Repository**

```bash
git clone https://github.com/sinmkeskin/Emotion-Aware-Facial-Recognition-and-Analysis-System
cd Emotion-Aware-Facial-Recognition-and-Analysis-System
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the App**

```bash
streamlit run .\app.py
```

## 🔍 Features

- Face detection using computer vision
- Emotion classification (e.g. happy, sad, angry)
- Database logging of detected emotions
- Dynamic responses based on emotion
- Extendable architecture for advanced analysis

## ⚠️ Notes

- Ensure that required model files are placed in the `model/` directory.
- Face images for recognition should be stored in `known_faces/`.

## 📜 License

This project is licensed under the MIT License.
