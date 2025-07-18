import os
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dotenv import load_dotenv
import shutil
from datetime import datetime
import sys
from datetime import timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Try to import TextProcessor
try:
    from text_processing import TextProcessor
except ImportError as e:
    st.error(f"Error importing TextProcessor: {e}")
    TextProcessor = None  # Will use mock processing if import fails

# Load environment variables
load_dotenv()

# Set up directories
UPLOAD_DIR = Path("uploads")
CV_DIR = UPLOAD_DIR / "cvs"
JOBS_DIR = UPLOAD_DIR / "job_descriptions"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, CV_DIR, JOBS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize TextProcessor
text_processor = None
if TextProcessor:
    try:
        text_processor = TextProcessor()
    except Exception as e:
        st.warning(f"Could not initialize TextProcessor: {e}")

# Set page config
st.set_page_config(
    page_title="OffHeadHunter - AI-Powered Job Matching",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #f0f2f6;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #e3f2fd;
        margin-right: auto;
    }
    .file-uploader {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file, directory: Path) -> Path:
    """Save uploaded file to the specified directory with a unique filename."""
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = directory / filename
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def extract_cv_info(file_path: Path) -> Dict[str, Any]:
    """Extract information from CV using TextProcessor."""
    if not text_processor:
        return extract_cv_info_mock(file_path)
    
    try:
        # Process the document
        chunks = text_processor.process_document(file_path)
        
        # Combine all chunks for analysis
        full_text = "\n\n".join([chunk['text'] for chunk in chunks])
        
        # Simple keyword extraction (can be enhanced with more sophisticated NLP)
        skills = extract_skills(full_text)
        experience = extract_experience(full_text)
        education = extract_education(full_text)
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "processed_at": datetime.now().isoformat(),
            "skills": skills,
            "experience_years": experience,
            "education": education,
            "full_text": full_text[:5000] + "..." if len(full_text) > 5000 else full_text,
            "chunks": [{"text": c['text'][:500] + "..." if len(c['text']) > 500 else c['text'], 
                        "num_tokens": c['num_tokens']} for c in chunks[:3]]  # Only include first 3 chunks for display
        }
        
    except Exception as e:
        st.error(f"Error processing CV: {str(e)}")
        return extract_cv_info_mock(file_path)  # Fall back to mock data

def extract_skills(text: str) -> List[str]:
    """Extract skills from text using simple keyword matching."""
    # This is a simple implementation - consider using NER or a skills database
    skills_keywords = [
        "python", "machine learning", "data analysis", "sql", "pandas", 
        "numpy", "pytorch", "tensorflow", "scikit-learn", "deep learning",
        "natural language processing", "nlp", "computer vision", "cv",
        "docker", "kubernetes", "aws", "gcp", "azure", "cloud computing",
        "git", "github", "gitlab", "ci/cd", "devops", "agile", "scrum",
        "javascript", "react", "node.js", "html", "css", "typescript"
    ]
    
    found_skills = set()
    text_lower = text.lower()
    
    for skill in skills_keywords:
        if skill in text_lower:
            found_skills.add(skill.title())
    
    return sorted(list(found_skills)) if found_skills else ["Python", "Data Analysis", "Machine Learning"]  # Default skills

def extract_experience(text: str) -> int:
    """Extract years of experience from text."""
    # Simple regex to find years of experience
    import re
    
    # Look for patterns like "X years of experience"
    match = re.search(r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)\s+(?:of)?\s+experience', text, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 30)  # Cap at 30 years
    
    # Default to 3 years if not found
    return 3

def extract_education(text: str) -> List[str]:
    """Extract education information from text."""
    # Look for common degree patterns
    degrees = []
    
    if re.search(r'\b(?:bachelor|b\.?s\.?|b\.?a\.?|b\.?eng\.?)\b', text, re.IGNORECASE):
        degrees.append("Bachelor's Degree")
    if re.search(r'\b(?:master|msc?|m\.?a\.?|m\.?eng\.?|m\.?ba\.?)\b', text, re.IGNORECASE):
        degrees.append("Master's Degree")
    if re.search(r'\b(?:ph\.?d|doctorate|dphil)\b', text, re.IGNORECASE):
        degrees.append("PhD")
    
    return degrees if degrees else ["Bachelor's Degree"]  # Default education

def extract_cv_info_mock(file_path: Path) -> Dict[str, Any]:
    """Mock function for CV info extraction when TextProcessor is not available."""
    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
        "processed_at": datetime.now().isoformat(),
        "skills": ["Python", "Machine Learning", "Data Analysis", "SQL", "Pandas", "Numpy"],
        "experience_years": 3,
        "education": ["Bachelor's in Computer Science"],
        "full_text": "[Content of the CV would appear here]",
        "chunks": [
            {"text": "[First part of the CV...]", "num_tokens": 450},
            {"text": "[Second part of the CV...]", "num_tokens": 500}
        ]
    }

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = [
        {"role": "assistant", "content": "👋 Welcome to OffHeadHunter! I'll help you find your dream job. Let's start by uploading your CV."}
    ]
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        "cv_info": None,
        "job_search_params": {},
        "job_results": []
    }

# Sidebar for job search parameters
with st.sidebar:
    st.title("🔍 Job Search Parameters")
    
    # CV upload section
    st.markdown("---")
    st.subheader("📄 Upload Your CV")
    uploaded_file = st.file_uploader(
        "Upload your CV (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="cv_uploader"
    )
    
    if uploaded_file is not None and not st.session_state.user_data.get("cv_info"):
        try:
            # Save the uploaded file
            cv_path = save_uploaded_file(uploaded_file, CV_DIR)
            
            # Process the CV
            with st.spinner("Analyzing your CV..."):
                cv_info = extract_cv_info(cv_path)
                st.session_state.user_data["cv_info"] = cv_info
                
                # Add CV summary to conversation
                skills_text = ", ".join(cv_info["skills"][:5])
                if len(cv_info["skills"]) > 5:
                    skills_text += f" and {len(cv_info['skills']) - 5} more"
                
                summary = f"✅ Successfully processed your CV! I found {len(cv_info['skills'])} skills including {skills_text}."
                if cv_info.get("experience_years"):
                    summary += f" You have approximately {cv_info['experience_years']} years of experience."
                
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": summary
                })
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing CV: {str(e)}")
    
    # Only show job search parameters if CV is uploaded
    if st.session_state.user_data.get("cv_info"):
        st.markdown("---")
        st.subheader("🔍 Search Parameters")
        
        # Job title input
        job_title = st.text_input(
            "Desired Job Title", 
            value=st.session_state.user_data.get("job_search_params", {}).get("job_title", "")
        )
        
        # Salary range
        col1, col2 = st.columns(2)
        with col1:
            min_salary = st.number_input(
                "Minimum Salary (€)", 
                min_value=0, 
                value=st.session_state.user_data.get("job_search_params", {}).get("min_salary", 30000),
                step=1000
            )
        with col2:
            max_salary = st.number_input(
                "Maximum Salary (€)", 
                min_value=0, 
                value=st.session_state.user_data.get("job_search_params", {}).get("max_salary", 70000),
                step=1000
            )
        
        # Location and work type
        location = st.text_input(
            "Preferred Location", 
            value=st.session_state.user_data.get("job_search_params", {}).get("location", "Spain")
        )
        
        work_type = st.multiselect(
            "Work Type",
            ["Full-time", "Part-time", "Contract", "Internship", "Remote"],
            default=st.session_state.user_data.get("job_search_params", {}).get("work_type", ["Full-time", "Remote"])
        )
        
        # Job portals
        job_portals = st.multiselect(
            "Job Portals",
            ["LinkedIn", "Indeed", "InfoJobs", "Glassdoor"],
            default=st.session_state.user_data.get("job_search_params", {}).get("job_portals", ["InfoJobs"])
        )
        
        # Save search parameters
        st.session_state.user_data["job_search_params"] = {
            "job_title": job_title,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "location": location,
            "work_type": work_type,
            "job_portals": job_portals
        }
        
        # Search button
        if st.button("🔍 Start Job Search", use_container_width=True, type="primary"):
            # Add search message to conversation
            search_message = f"🔍 Searching for {job_title} jobs in {location} with salary range €{min_salary:,}-€{max_salary:,}"
            st.session_state.conversation.append({"role": "user", "content": search_message})
            
            # Simulate job search (will be replaced with actual search)
            with st.spinner("Searching for matching jobs..."):
                time.sleep(2)  # Simulate search time
                
                # TODO: Integrate with job_search_agent.py and scraper.py
                # For now, use mock data
                mock_jobs = []
                companies = ["TechCorp", "InnovateSoft", "DataSystems", "WebCrafters", "AI Solutions"]
                
                for i in range(3):
                    company = companies[i % len(companies)]
                    is_remote = i % 2 == 0
                    job_location = "Remote" if is_remote else location
                    
                    # Calculate match score based on CV skills
                    match_score = 80 + (i * 5)  # Base score + variation
                    
                    # Adjust score based on job title match with CV
                    cv_skills = st.session_state.user_data["cv_info"].get("skills", [])
                    title_boost = sum(1 for skill in ["Senior", "Lead", "Principal"] if skill.lower() in job_title.lower()) * 5
                    match_score = min(100, match_score + title_boost)
                    
                    job = {
                        "id": f"job_{i}_{int(time.time())}",
                        "title": f"{job_title} {'Specialist' if i % 2 == 0 else 'Engineer'}",
                        "company": company,
                        "location": job_location,
                        "salary": f"€{min_salary + (i * 5000):,} - €{max_salary - (i * 2000):,}",
                        "match_score": match_score,
                        "description": (
                            f"Looking for an experienced {job_title} to join our team at {company}. "
                            f"{'Remote work available.' if is_remote else 'On-site position in ' + location + '.'} "
                            f"Required skills: {', '.join(cv_skills[:3])}."
                        ),
                        "requirements": [
                            f"{st.session_state.user_data['cv_info'].get('experience_years', 3) + (i-1)} years of experience in a similar role",
                            f"Strong knowledge of {', '.join(cv_skills[:2]) if cv_skills else 'relevant technologies'}",
                            f"{st.session_state.user_data['cv_info'].get('education', ['Bachelor\'s degree'])[0]} or equivalent experience"
                        ],
                        "url": "#",
                        "source": job_portals[i % len(job_portals)] if job_portals else "InfoJobs",
                        "posted_date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "is_remote": is_remote
                    }
                    mock_jobs.append(job)
                    
                    # Save job description as JSON
                    job_file = JOBS_DIR / f"{job['id']}.json"
                    with open(job_file, 'w') as f:
                        json.dump(job, f, indent=2)
                
                # Sort jobs by match score (highest first)
                mock_jobs.sort(key=lambda x: x["match_score"], reverse=True)
                
                st.session_state.user_data["job_results"] = mock_jobs
                
                # Add results to conversation
                top_job = mock_jobs[0] if mock_jobs else {}
                results_message = (
                    f"✅ Found {len(mock_jobs)} matching jobs. "
                    f"Top match: {top_job.get('title', 'N/A')} at {top_job.get('company', 'a company')} "
                    f"with a {top_job.get('match_score', 0)}% match!"
                )
                st.session_state.conversation.append({"role": "assistant", "content": results_message})
                st.rerun()

# Main content area
st.title("💼 OffHeadHunter - AI-Powered Job Matching")

# Chat interface
st.subheader("🤖 Chat with OffHeadHunter")

# Display conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display CV info if available
cv_info = st.session_state.user_data.get("cv_info")
if cv_info:
    with st.expander("📋 View CV Information", expanded=False):
        st.write(f"**File:** {cv_info['file_name']} ({cv_info['file_size'] / 1024:.1f} KB)")
        
        # Skills section
        st.markdown("### Skills")
        if cv_info.get("skills"):
            st.write(" ".join([f"<span class='skill-tag'>{skill}</span>" for skill in cv_info["skills"]]), 
                    unsafe_allow_html=True)
        
        # Experience and Education
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Experience")
            st.write(f"**Years:** {cv_info.get('experience_years', 'N/A')}")
        
        with col2:
            st.markdown("### Education")
            for edu in cv_info.get("education", ["Not specified"]):
                st.write(f"• {edu}")
        
        # Document preview
        st.markdown("### Document Preview")
        st.text_area("Extracted Text", 
                    cv_info.get("full_text", "No text content available"),
                    height=200)
        
        # Show chunking info if available
        if cv_info.get("chunks"):
            with st.expander("View Document Chunks"):
                for i, chunk in enumerate(cv_info["chunks"][:3]):  # Show first 3 chunks
                    st.markdown(f"**Chunk {i+1}** ({chunk['num_tokens']} tokens)")
                    st.text(chunk["text"])
                    st.markdown("---")

# Display job results if available
job_results = st.session_state.user_data.get("job_results", [])
if job_results:
    st.markdown("---")
    st.subheader(f"🔍 {len(job_results)} Matching Jobs")
    
    for job in job_results:
        with st.expander(
            f"{job['title']} at {job['company']} • "
            f"{job['location']} • "
            f"**{job['match_score']}% Match** • "
            f"{job['salary']}",
            expanded=False
        ):
            # Job header with company and match score
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"#### {job['title']}")
                st.markdown(f"**{job['company']}** • {job['location']} • {job['posted_date']}")
            
            with col2:
                st.metric("Match Score", f"{job['match_score']}%")
            
            # Job details
            st.markdown("##### Overview")
            st.write(job['description'])
            
            st.markdown("##### Requirements")
            for req in job.get('requirements', []):
                st.write(f"• {req}")
            
            # Action buttons
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Apply Now", key=f"apply_{job['id']}"):
                    st.session_state.conversation.append({
                        "role": "user",
                        "content": f"I want to apply for the {job['title']} position at {job['company']}."
                    })
                    st.rerun()
            
            with col2:
                st.markdown(f"[View on {job['source']}]({job['url']})" if job['url'] != "#" else "*Source link not available*")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    # TODO: Process message with job_search_agent.py
    # For now, just echo the message
    response = f"You said: {prompt}"
    st.session_state.conversation.append({"role": "assistant", "content": response})
    
    # Rerun to update the chat
    st.rerun()

# Add some space at the bottom
st.markdown("""
    <div style="margin-top: 5rem;"></div>
    <div style="text-align: center; color: #666;">
        <p>OffHeadHunter - Find your dream job with AI assistance</p>
    </div>
""", unsafe_allow_html=True)
