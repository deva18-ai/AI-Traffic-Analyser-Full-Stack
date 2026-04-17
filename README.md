# 🚦 AI Traffic Analyzer (Full Stack)

<p align="center">
  <b>AI-powered traffic analysis system using YOLOv8, FastAPI, and modern web technologies</b>
</p>

---

## 📌 Overview

AI Traffic Analyzer is a full-stack application that detects, analyzes, and provides insights on traffic using computer vision and AI.

It allows users to upload images or videos and automatically:

* Detect vehicles 🚗
* Count traffic density 📊
* Generate analytics 📈
* Store and retrieve results 📂

---

## 🚀 Features

* 🔍 **Vehicle Detection** using YOLOv8
* 📊 **Traffic Analysis Dashboard**
* 📁 Upload images & videos
* 🧠 AI-based object recognition
* 📈 Statistics & insights API
* 🌐 Full-stack integration (Frontend + Backend)
* ⚡ FastAPI backend for high performance

---

## 🛠️ Tech Stack

### 🔹 Backend

* Python
* FastAPI
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy

### 🔹 Frontend

* HTML / CSS / JavaScript
* (Optional: React / Next.js if used)

### 🔹 Tools

* Git & GitHub
* Vercel (Frontend Deployment)
* Render (Backend Deployment)

---

## 📂 Project Structure

```
AI-Traffic-Analyser-Full-Stack/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── models/
│   └── utils/
│
├── frontend/
│   ├── index.html
│   ├── scripts/
│   └── styles/
│
├── uploads/
├── results/
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/deva18-ai/AI-Traffic-Analyser-Full-Stack.git
cd AI-Traffic-Analyser-Full-Stack
```

---

### 2️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload
```

---

### 3️⃣ Frontend Setup

Simply open:

```
frontend/index.html
```

OR deploy using Vercel.

---

## 📡 API Endpoints

| Method | Endpoint             | Description        |
| ------ | -------------------- | ------------------ |
| POST   | /api/upload          | Upload image/video |
| POST   | /api/analyze-traffic | Analyze traffic    |
| GET    | /api/results         | Get all results    |
| GET    | /api/results/{id}    | Get single result  |
| GET    | /api/stats           | Traffic statistics |

---

## 🧠 How It Works

1. User uploads media 📤
2. Backend processes using YOLOv8 🤖
3. Vehicles are detected and counted 🚗
4. Results stored and returned 📊
5. Frontend displays analytics 📈

---

## 🌍 Deployment

### Frontend:

* Deploy on **Vercel**

### Backend:

* Deploy on **Render**

---

## 📸 Screenshots

*Add your UI screenshots here*

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Devavardhan Mohanraj**

* GitHub: https://github.com/deva18-ai

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
