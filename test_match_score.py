"""
Test Script for Calculating Match Scores between CVs and Job Offers

This script allows you to manually input a CV and job offers, then calculates
the match score between them using the semantic matching functionality.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timezone

# Add the src directory to the Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / 'src'
sys.path.append(str(project_root))
sys.path.append(str(src_path))

# Import the matching function from test_semantic_matching.py
from test_semantic_matching import find_similar_jobs, preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_input():
    """Get CV and job offers input from the user."""
    print("=== CV and Job Offer Match Score Calculator ===")
    
    # Get CV text
    print("\n=== Enter your CV text (press Enter twice when finished):")
    cv_lines = []
    while True:
        try:
            line = input()
            if line == "" and cv_lines and cv_lines[-1] == "":
                break
            cv_lines.append(line)
        except EOFError:
            break
    cv_text = "\n".join(cv_lines).strip()
    
    # Get job offers
    job_offers = []
    print("\n=== Enter job offers (press Enter twice after each offer, and twice when finished):")
    
    while True:
        print(f"\nJob Offer #{len(job_offers) + 1} (leave empty to finish):")
        job_lines = []
        while True:
            try:
                line = input()
                if line == "" and (not job_lines or job_lines[-1] == ""):
                    break
                job_lines.append(line)
            except EOFError:
                break
        
        job_text = "\n".join(job_lines).strip()
        if not job_text:
            break
            
        job_offers.append({
            "title": f"Job Offer #{len(job_offers) + 1}",
            "description": job_text,
            "scraped_at": datetime.now(timezone.utc).isoformat()
        })
    
    return cv_text, job_offers

def calculate_match_scores(cv_text, job_offers):
    """Calculate match scores between CV and job offers."""
    print("\n=== Calculating Match Scores ===")
    
    # For each job offer, calculate the match score with the CV
    results = []
    for job in job_offers:
        # Combine title and description for matching
        job_text = f"{job['title']}\n{job['description']}"
        
        # Calculate match score using the existing function
        match_result = find_similar_jobs(
            cv_text=cv_text,
            job_offers=[job],  # Process one job at a time
            include_keywords=True
        )
        
        if match_result and 'matches' in match_result and match_result['matches']:
            match_score = match_result['matches'][0].get('score', 0) * 100  # Convert to percentage
            results.append((job['title'], match_score, job['description']))
    
    # Sort results by match score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def display_results(cv_text, results):
    """Display the match score results."""
    print("\n=== CV Text ===")
    print(cv_text[:500] + ("..." if len(cv_text) > 500 else ""))
    
    print("\n=== Match Scores ===")
    if not results:
        print("No valid matches found.")
        return
    
    for i, (title, score, description) in enumerate(results, 1):
        print(f"\n{i}. {title}")
        print(f"   Match Score: {score:.1f}%")
        print(f"   Description: {description[:200]}{'...' if len(description) > 200 else ''}")

def main():
    """Main function to run the match score calculator."""
    try:
        # Get user input
        cv_text, job_offers = get_user_input()
        
        if not cv_text:
            print("Error: CV text cannot be empty.")
            return
            
        if not job_offers:
            print("Error: At least one job offer is required.")
            return
        
        # Calculate match scores
        results = calculate_match_scores(cv_text, job_offers)
        
        # Display results
        display_results(cv_text, results)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
