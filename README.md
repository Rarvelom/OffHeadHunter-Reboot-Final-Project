# OffHeadHunter – RebootAcademy Final Project

OffHeadHunter is an AI-powered job search assistant designed to personalize and streamline the recruitment journey. Through a conversational interface, it captures user preferences, analyzes CVs, and matches candidates with relevant job offers—culminating with intelligent CV adaptation to maximize relevance for specific job opportunities.

This project has been developed collaboratively as part of a Data Science and Artificial Intelligence bootcamp final challenge at RebootAcademy.

---

## MVP Description

### Key Features
- Chatbot-based interaction for job search preference capture
- CV upload and vectorization
- Match Score system between CV and job offers
- Web scraping of listings from selected job portals (primarily InfoJobs)
- Adapted CV generation using LLMs
- Semantic processing with embeddings for higher precision

### Workflow Summary
1. User interacts with the chatbot to define job preferences
2. User uploads their CV to the system
3. System extracts, vectorizes, and analyzes the CV
4. Relevant job offers are scraped and processed
5. Match scores are calculated for each job offer
6. An adapted version of the CV is generated, optimized for the selected job offer

---

## System Architecture

### Main Components
- **Chatbot Module**: Conversational interface for capturing preferences
- **Document Processor**: Extraction, chunking, and vectorization of CVs and job offers
- **Matching Engine**: Semantic comparison system based on embeddings
- **CV Adapter**: Generation of personalized CVs using Gemini AI
- **Integrated Pipeline**: Orchestration of the complete workflow

### Tech Stack
| Component             | Technologies                                     |
|-----------------------|------------------------------------------------|
| Backend / Logic       | Python, FastAPI                                 |
| Databases             | MongoDB, Qdrant (for vector search)             |
| AI Processing         | Gemini API, SentenceTransformers, BGE Embeddings|
| Extraction & Scraping | Unstructured, Custom Web Scraping               |
| Document Generation   | ReportLab, Markdown                             |
| Frontend / UI         | Streamlit                                       |

---

## Installation & Usage

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
```

### Environment Variables
Create a `.env` file with the following variables:
```
MONGODB_URI=mongodb://localhost:27017/
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Running the Application
```bash
streamlit run app.py
```

---

## Future Development
- Integration with additional job portals (LinkedIn, Indeed, etc.)
- Kanban dashboard for application tracking
- AI interview coach with TTS/STT for mock interviews
- Enhanced web interface with HTML/CSS/JS
- Generation of personalized cover letters
- Proactive recommendation system and metrics of success

---

## Team
- Alberto Domínguez González
- Azhara García Asencio
- Ricardo Arvelo

## License
This project is under the MIT License.
