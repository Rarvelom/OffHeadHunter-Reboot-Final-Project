# OffHeadHunter – RebootAcademy Final Project

OffHeadHunter is an AI-powered job search assistant designed to personalize and streamline the recruitment journey. Through a conversational interface, it captures user preferences, analyzes CVs, and matches candidates with relevant job offers—culminating in a Kanban-style dashboard that tracks the application process end-to-end.

This project is being developed collaboratively over a 3-week sprint cycle as part of a bootcamp final challenge.

---

##  MVP Overview

###  Key Features
- Chatbot-based interaction for job search input
- CV upload and vectorization
- Match Score between CV and job offers
- Scraped job listings from selected portals (LinkedIn, InfoJobs, Indeed)
- Adapted CV generation with LLMs
- Application tracking via visual dashboard (Kanban flow)

###  Workflow Summary
1. User registers and uploads CV
2. Bot gathers search preferences
3. System scrapes portals + vectorizes CV
4. Top offers are selected based on match score
5. Dashboard organizes applications by status

---

## Tech Stack

| Area                | Tools / Frameworks                                |
|---------------------|----------------------------------------------------|
| Backend / Logic     | Python, FastAPI, N8N, Scrapy                       |
| Database            | MongoDB (planned), Qdrant for vector search        |
| Frontend / UI       | Streamlit, HTML/CSS/JS, Bolt.io, Lovable           |
| AI Components       | Gemini API, HuggingFace Models, RAG + LLMs         |
| TTS/STT (Optional)  | Whisper + Coqui / Google TTS (if time permits)     |

---

##  Getting Started

### Prerequisites
- Python 3.10+
- MongoDB local or Atlas
- Virtual environment (recommended)
- Git + GitHub account

### Setup
```bash
git clone https://github.com/Rarvelom/OffHeadHunter-Reboot-Final-Project.git
cd OffHeadHunter-Reboot-Final-Project
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
