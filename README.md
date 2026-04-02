# 🌍 LinguaDetect — AI Language Detection

A full-stack web application that identifies the language of any text using a deep learning model Naive Bayes+Logistic Regression trained on 17 languages.

---

## 📁 Project Structure

```
langdetect/
├── backend/
│   └── main.py              ← FastAPI application
├── frontend/
│   └── index.html           ← Beautiful colorful UI
├── model/
│   └── language_detector.keras  ← Your trained Keras model
├── requirements.txt
├── run.sh                   ← One-click startup
└── README.md
```

---

## 🚀 Quick Start

### 1. Place your model
```bash
# Copy your trained model into the model/ folder:
cp /path/to/language_detector.keras model/language_detector.keras
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
bash run.sh
# OR manually:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in browser
```
http://localhost:8000
```

---

## 🌐 Supported Languages (17)

| Language   | Flag | Native Name |
|------------|------|-------------|
| Arabic     | 🇸🇦 | العربية      |
| Danish     | 🇩🇰 | Dansk        |
| Dutch      | 🇳🇱 | Nederlands   |
| English    | 🇬🇧 | English      |
| French     | 🇫🇷 | Français     |
| German     | 🇩🇪 | Deutsch      |
| Greek      | 🇬🇷 | Ελληνικά     |
| Hindi      | 🇮🇳 | हिन्दी        |
| Italian    | 🇮🇹 | Italiano     |
| Kannada    | 🇮🇳 | ಕನ್ನಡ          |
| Malayalam  | 🇮🇳 | മലയാളം        |
| Portugese  | 🇵🇹 | Português    |
| Russian    | 🇷🇺 | Русский      |
| Spanish    | 🇪🇸 | Español      |
| Swedish    | 🇸🇪 | Svenska      |
| Tamil      | 🇮🇳 | தமிழ்         |
| Turkish    | 🇹🇷 | Türkçe       |

---

## 🔌 API Endpoints

### `POST /detect`
Detect the language of a text.

**Request:**
```json
{ "text": "Hello, how are you?" }
```

**Response:**
```json
{
  "detected": {
    "language": "English",
    "confidence": 99.87,
    "flag": "🇬🇧",
    "native_name": "English",
    "color": "#6366F1",
    "rank": 1
  },
  "top_predictions": [ ...5 items... ],
  "char_count": 20,
  "word_count": 4
}
```

### `GET /languages`
Returns all 17 supported languages with metadata.

### `GET /health`
Health check endpoint.

### `GET /docs`
Interactive Swagger API documentation.

---

## 🧠 Model Details

- **Architecture:** CharCNN (Character-level Convolutional Neural Network)
- **Input:** Raw UTF-8 byte sequences (max 300 bytes)
- **Layers:** Embedding → 3× Conv1D + BatchNorm + MaxPool → GlobalMaxPool → Dense
- **Output:** Softmax over 17 language classes
- **Training Dataset:** [Kaggle Language Detection](https://www.kaggle.com/datasets/basilb2s/language-detection)
- **Expected Accuracy:** 99%+

---

## ⌨️ Keyboard Shortcut

- **Ctrl + Enter** — Detect language

---

## 🛠 Tech Stack

| Layer    | Technology              |
|----------|-------------------------|
| Frontend | Vanilla HTML/CSS/JS     |
| Backend  | FastAPI + Uvicorn       |
| ML       | TensorFlow / Keras      |
| Fonts    | Syne + DM Sans (Google) |
