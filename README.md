# Resume Scorer Agentic AI

## Overview

This project is a resume scoring and matching system built with FastAPI, SQLite, ChromaDB, and Google Gemini embeddings.

It allows a user to upload a resume PDF, extract and parse the resume text, save structured resume data and embeddings, and then match a job description to the best resumes.

## Project Flow

                 Resume Scorer Agentic AI - Flow

                         User
                          |
                          |
                    Upload Resume PDF
                          |
                          v
              +------------------------+
              |     FastAPI API        |
              |   POST /upload-resume  |
              +------------------------+
                          |
                          v
                 Extract PDF Text
                    (PyPDF2)
                          |
                          v
              Store Raw Resume Data
                    (SQLite)
                          |
                          |
                          v
             Parse Resume Information
          (Gemini LLM - JSON Extraction)
                          |
                          v
          Structured Resume Data
     (Email, Skills, Experience, Projects)
                          |
                          v
                  Save Parsed Data
                    (SQLite)
                          |
                          v
              Create Text Embeddings
             (Gemini Embedding Model)
                          |
                          v
              Store Embeddings + Metadata
                    (ChromaDB)
                          |
                          |
        ----------------------------------
        |                                |
        v                                v

 GET /resume-metadata/{id}          POST /match-job

        |                                |
        v                                v

 Fetch Resume Vectors             User Enters Job Description
        |                                |
        v                                v

 Return Resume Sections            Create Job Embedding
 (JSON Response)                         |
                                         v
                              Search Similar Vectors
                                  (ChromaDB)
                                         |
                                         v
                              Calculate Similarity Score
                                         |
                                         v
                              Rank Matching Resumes
                                         |
                                         v
                              Return Best Candidates


                          |
                          v

                 Dashboard UI
        Upload | View Resume | Match Job | Results

1. **Resume upload**
   - The user uploads a PDF via `POST /upload-resume`.
   - `PyPDF2` extracts the text from the PDF.
   - The extracted text is stored in a local SQLite database using SQLAlchemy.

2. **Resume parsing**
   - A resume is parsed with `GET /parse-resume/{resume_id}`.
   - The parsed resume text is sent to a Gemini LLM model through `save_text_embedding.parse_resume_text_to_dict(...)`.
   - The model returns structured JSON with keys like `email`, `skills`, `experience`, and `projects`.
   - Parsed data is saved back into the database.

3. **Embedding creation**
   - Parsed resume sections are converted into embeddings using Google Gemini embeddings.
   - Embeddings, metadata, and the original section text are stored in ChromaDB.

4. **Job matching**
   - The user submits a job description to `POST /match-job`.
   - The job description is converted to an embedding.
   - ChromaDB is queried for similar resume sections.
   - Results are aggregated by resume email and returned as ranked matches.

5. **Resume metadata lookup**
   - `GET /resume-metadata/{resume_id}` returns stored embedding sections for a resume.

6. **Dashboard UI**
   - The project includes a web dashboard served at `GET /`.
   - The UI supports uploading resumes, listing saved resumes, and matching job descriptions.

## Technologies Used

- Python
- FastAPI
- Uvicorn
- SQLite + SQLAlchemy
- ChromaDB
- Google Gemini embeddings and generative models
- `langchain-google-genai`
- `python-dotenv`
- PyPDF2
- HTML + JavaScript for the dashboard UI

## Files

- `main.py` - FastAPI server and endpoint definitions
- `database.py` - SQLAlchemy database setup
- `models.py` - Resume data model
- `save_text_embedding.py` - Text embedding and parsing logic
- `vector_embedding_db.py` - ChromaDB storage helpers
- `static/index.html` - Front-end dashboard
- `requirements.txt` - Python dependencies

## Run

```bash
cd Resume-Scorer-Agentic-AI
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```
