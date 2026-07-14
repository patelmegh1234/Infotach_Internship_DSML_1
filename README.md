<h1 align="center">🎙️ AcousticSpace</h1>

<p align="center">
  <b>Deepfake Detection via Room Impulse Response (RIR)</b><br>
  <i>AI-Powered Audio Forensics Platform | Internship Project</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Status-In%20Progress-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Team-4%20Members-blueviolet?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Librosa-Audio%20Processing-8E44AD?style=flat-square" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformer%20Model-FFB000?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-Deployment-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-UI%20Development-3178C6?style=flat-square&logo=typescript&logoColor=white" />
</p>

***

## 📖 About This Repository

**AcousticSpace** is a full-stack AI project focused on detecting **deepfake audio** using **Room Impulse Response (RIR)**, environmental acoustics, and temporal speech-breathing consistency. Instead of relying only on voice biometrics, the system evaluates whether the voice realistically matches the surrounding recording environment.

> 👨‍💻 **Team Lead:** Megh  
> 👥 **Team Members:** Megh, Dimple, Shubhangi, Fathima  
> 🏢 **Project Type:** Internship Team Project  
> 🎯 **Domain:** Audio Forensics & Deepfake Detection  

***

## 🧠 Problem Statement

Conventional deepfake audio detectors mainly focus on vocal artifacts such as robotic texture or unnatural intonation. As generative AI improves, these surface-level cues become easier to hide.

**AcousticSpace** addresses this by analyzing the **physics behind the sound** — especially room reflections, reverberation, and cadence consistency — to identify audio that may be artificially generated.

***

## 🚀 Use Case

A security analyst uploads a suspicious audio clip to the platform. AcousticSpace processes the signal, extracts low-level acoustic characteristics, and checks whether the **voice, breathing rhythm, and room response** are physically aligned.

If the background environment and generated voice do not match, the system flags the recording as potentially synthetic and presents the result through an interactive dashboard.

***

## 🗂️ Project Overview

| Module | Responsibility | Tech Stack |
|:---:|---|---|
| **1** | Audio Processing Pipeline | Python, Librosa, NumPy |
| **2** | Deepfake Classification Model | PyTorch, Hugging Face AST |
| **3** | API & Inference Layer | FastAPI, Uvicorn, Pydantic |
| **4** | Analyst Dashboard | React, TypeScript, WaveSurfer.js |

***

## 📂 Repository Structure

```bash
AcousticSpace/
│
├── README.md                         ← Main project overview
├── .gitignore                        ← Git ignore rules
├── requirements.txt                  ← Python dependencies
│
├── backend/                          ← FastAPI backend services
│   ├── app/
│   ├── api/
│   ├── services/
│   ├── models/
│   └── main.py
│
├── ml/                               ← Machine learning and preprocessing
│   ├── data/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   └── notebooks/
│
├── frontend/                         ← React dashboard
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── package.json
│
├── docker/                           ← Docker configuration files
├── tests/                            ← Testing modules
└── docs/                             ← Technical documentation
```

***

## ⚙️ Core Features

- **RIR-based deepfake detection** using room acoustic signatures
- **Librosa-powered preprocessing** for spectrogram and environmental feature extraction
- **Transformer-based classifier** for robust audio authenticity analysis
- **FastAPI backend** for low-latency model serving
- **React dashboard** for upload, waveform display, and result visualization
- **Confidence scoring** for investigation support
- **Modular architecture** for easy scaling and deployment

***

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Programming Languages** | Python, TypeScript |
| **Audio Processing** | Librosa, NumPy, SciPy |
| **Machine Learning** | PyTorch, Hugging Face Transformers |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | React, TypeScript |
| **Visualization** | WaveSurfer.js |
| **Deployment** | Docker, CI/CD |

***

## 📅 Development Roadmap

### Week 1: Foundation Setup 🎯
> **Phase:** Core Setup & Data Preparation

- Set up FastAPI backend
- Curate dataset for training and testing
- Build Librosa preprocessing pipeline
- Create React app structure and audio upload UI

### Week 2: Baseline Model & Visualization 📊
> **Phase:** Baseline Intelligence

- Train initial CNN/Transformer baseline model
- Validate extracted acoustic features
- Integrate waveform visualization on frontend
- Review file upload handling and dashboard flow

### Week 3: Advanced Detection Logic 🧠
> **Phase:** Model Refinement

- Fine-tune Hugging Face Audio Transformer
- Implement breathing cadence analysis
- Build confidence score result panel
- Highlight suspicious audio regions in the dashboard

### Week 4: Deployment & Finalization 🚀
> **Phase:** Production Readiness

- Dockerize backend and ML pipeline
- Optimize inference latency
- Add CI/CD workflows
- Final UI/UX improvements and history tracking

***

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Git**
- **Docker** (optional)

### Clone the Repository

```bash
git clone https://github.com/your-username/AcousticSpace.git
cd AcousticSpace
```

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

***

## 👥 Team Contribution Plan

| Member | Assigned Area | Key Ownership |
|---|---|---|
| **Megh Patel** | Backend + Integration + Team Leadership | Repo setup, FastAPI integration, review, deployment support |
| **Dimple** | Audio Processing | Dataset setup, preprocessing, RIR feature extraction |
| **Shubhangi** | Machine Learning | Baseline model, transformer fine-tuning, evaluation |
| **Fathima** | Frontend Dashboard | React UI, waveform visualization, results interface |

***

## 🌟 Skills Demonstrated

```text
✅ Audio Signal Processing             ✅ Deepfake Detection Research
✅ Room Impulse Response Analysis      ✅ Transformer-Based Classification
✅ FastAPI Backend Development         ✅ React Dashboard Development
✅ API Integration                     ✅ Model Inference Workflow
✅ Team Collaboration & Git Workflow   ✅ Docker & Deployment Basics
✅ End-to-End AI Product Development   ✅ Technical Documentation
```

***

## 🔀 Git Workflow

To ensure transparent contribution tracking, each team member should work on a dedicated branch and raise pull requests for review.

**Recommended branches:**
- `main` → stable branch
- `dev` → integration branch
- `feature/audio-pipeline`
- `feature/model-training`
- `feature/api-inference`
- `feature/frontend-dashboard`

***

## 🎯 Expected Outcome

AcousticSpace aims to deliver a robust and research-oriented audio deepfake detection platform that identifies synthetic speech by analyzing **physical acoustic mismatch**, not just vocal artifacts.

The final product includes a trained detection model, backend inference service, and an interactive dashboard for practical forensic analysis.

***

## 👤 Author & Team

**Megh Patel**  
Team Lead — AcousticSpace Internship Project

**Team Members:**  
- Megh Patel  
- Dimple  
- Shubhangi  
- Fathima

***

<p align="center">
  <i>⭐ If you found this project interesting, consider starring the repository.</i>
</p>
