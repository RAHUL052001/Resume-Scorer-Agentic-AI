import io
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Request
from typing import Optional
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import PyPDF2

from database import Base, SessionLocal, engine
from models import Resume
from save_text_embedding import parse_resume_text_to_dict, save_resume_embeddings, text_to_embedding
from vector_embedding_db import collection as chroma_collection

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Resume PDF Extractor API")

static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        return "\n".join(pages).strip()
    except Exception as exc:
        raise ValueError(str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "detail": "Resume PDF extractor is ready."}


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        extracted_text = extract_text_from_pdf_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {exc}")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="PDF contains no extractable text")

    # Check if resume with same content already exists
    existing_resume = db.query(Resume).filter(Resume.content == extracted_text).first()
    if existing_resume:
        return JSONResponse(
            status_code=200,
            content={
                "id": existing_resume.id,
                "filename": existing_resume.filename,
                "content": existing_resume.content,
                "upload_date": existing_resume.upload_date.isoformat(),
                "status": existing_resume.status,
                "message": "Resume with same content already exists in database",
                "is_duplicate": True,
            },
        )

    resume = Resume(
        filename=file.filename,
        content=extracted_text,
        upload_date=datetime.utcnow(),
        status="uploaded",
    )


    try:
        db.add(resume)
        db.commit()
        db.refresh(resume)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    return JSONResponse(
        status_code=201,
        content={
            "id": resume.id,
            "filename": resume.filename,
            "content": resume.content,
            "upload_date": resume.upload_date.isoformat(),
            "status": resume.status,
            "is_duplicate": False,
        },
    )


@app.get("/resumes")
def list_resumes(db: Session = Depends(get_db)):
    resumes = db.query(Resume).order_by(Resume.upload_date.desc()).all()
    return [
        {
            "id": item.id,
            "filename": item.filename,
            "upload_date": item.upload_date.isoformat() if item.upload_date else None,
            "status": item.status,
        }
        for item in resumes
    ]


@app.get("/resumes/{resume_id}")
def get_resume(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    return {
        "id": resume.id,
        "filename": resume.filename,
        "content": resume.content,
        "upload_date": resume.upload_date.isoformat() if resume.upload_date else None,
        "status": resume.status,
    }


@app.get("/parse-resume/{resume_id}")
def parse_resume(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    try:
        structured_resume = parse_resume_text_to_dict(resume.content, email=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM parse error: {exc}")

    # Save parsed data to SQLite
    try:
        resume.parsed_data = structured_resume
        resume.status = "parsed"
        db.commit()
        db.refresh(resume)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database save error: {exc}")

    # Attempt to save parsed sections as embeddings in ChromaDB
    try:
        saved_ids = save_resume_embeddings(structured_resume)
        embeddings_saved = True
    except ValueError as ve:
        # Likely missing email in structured data; return parsed result but don't save embeddings
        return {
            "id": resume.id,
            "filename": resume.filename,
            "parsed_resume": structured_resume,
            "embeddings_saved": False,
            "error": str(ve),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding save error: {exc}")

    return {
        "id": resume.id,
        "filename": resume.filename,
        "parsed_resume": structured_resume,
        "embeddings_saved": embeddings_saved,
        "saved_ids": saved_ids,
    }


@app.get("/resume-metadata/{resume_id}")
def get_resume_metadata(resume_id: int, db: Session = Depends(get_db)):
    """Retrieve metadata from ChromaDB for all sections of a resume."""
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if not resume.parsed_data or not resume.parsed_data.get("email"):
        return {
            "resume_id": resume_id,
            "status": "no_data",
            "message": "Resume has not been parsed yet or email is missing",
            "sections": [],
        }

    email = resume.parsed_data.get("email")

    # Query ChromaDB for all sections associated with this email
    try:
        results = chroma_collection.get(
            where={"email": email}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ChromaDB query error: {exc}")

    if not results or not results.get("ids"):
        return {
            "resume_id": resume_id,
            "email": email,
            "status": "no_data",
            "message": "No embedding data found for this resume in ChromaDB",
            "sections": [],
        }

    sections = []
    for idx, doc_id in enumerate(results.get("ids", [])):
        sections.append({
            "id": doc_id,
            "section": results["metadatas"][idx].get("section") if results.get("metadatas") else None,
            "email": results["metadatas"][idx].get("email") if results.get("metadatas") else None,
            "document_preview": results["documents"][idx][:100] + "..." if results.get("documents") and len(results["documents"][idx]) > 100 else results["documents"][idx] if results.get("documents") else None,
        })

    return {
        "resume_id": resume_id,
        "email": email,
        "status": "found",
        "sections": sections,
    }


@app.get("/")
async def docs_ui():
    index_path = Path(__file__).parent / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(index_path, media_type="text/html")


@app.post("/match-job")
async def match_job(request: Request, job_description: Optional[str] = None, top_k: int = 3, db: Session = Depends(get_db)):
    """Match a job description to the best resumes using ChromaDB embeddings.

    - `job_description`: raw text of the job description
    - `top_k`: number of top resumes to return
    """
    # Accept raw body (any content-type), form data, or JSON object

    # Try raw body first — this avoids FastAPI JSON parsing errors when client sends invalid JSON
    try:
        raw = await request.body()
        if raw:
            decoded = raw.decode("utf-8", errors="replace").strip()
            # If client sent a JSON string (e.g. "..."), strip surrounding quotes
            if decoded.startswith('"') and decoded.endswith('"') and len(decoded) >= 2:
                decoded = decoded[1:-1]
            job_description = decoded
    except Exception:
        job_description = None

    # If body empty, try form data (multipart/form-data or x-www-form-urlencoded)
    if not job_description:
        try:
            form = await request.form()
            if "job_description" in form:
                job_description = str(form.get("job_description"))
        except Exception:
            pass

    # If still empty, try JSON parsed by FastAPI (rare since we avoided Body(...))
    if not job_description:
        try:
            js = await request.json()
            if isinstance(js, dict) and "job_description" in js:
                job_description = js["job_description"]
            elif isinstance(js, str):
                job_description = js
        except Exception:
            pass

    if not job_description or not str(job_description).strip():
        raise HTTPException(status_code=400, detail="job_description is required")

    try:
        query_vector = text_to_embedding(job_description)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding generation error: {exc}")

    try:
        results = chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=50,
            include=["metadatas", "distances", "documents"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ChromaDB query error: {exc}")

    ids_list = results.get("ids", [[]])[0]
    distances_list = results.get("distances", [[]])[0]
    metadatas_list = results.get("metadatas", [[]])[0]
    documents_list = results.get("documents", [[]])[0]

    if not ids_list:
        return {"status": "no_data", "message": "No embeddings found in ChromaDB", "matches": []}

    # Aggregate best (minimum) distance per email
    best_by_email: dict = {}
    for idx, doc_id in enumerate(ids_list):
        dist = distances_list[idx] if idx < len(distances_list) else None
        meta = metadatas_list[idx] if idx < len(metadatas_list) else {}
        doc = documents_list[idx] if idx < len(documents_list) else None

        email = meta.get("email") if isinstance(meta, dict) else None
        section = meta.get("section") if isinstance(meta, dict) else None
        if not email:
            # skip entries without associated resume email
            continue

        entry = best_by_email.get(email)
        if entry is None or (dist is not None and dist < entry["distance"]):
            best_by_email[email] = {
                "distance": dist,
                "section": section,
                "doc_id": doc_id,
                "document": doc,
            }

    # Build match list with resume id lookup
    matches = []
    for email, info in best_by_email.items():
        distance = info.get("distance")
        score = None
        if distance is None:
            score = None
        else:
            # convert cosine distance to similarity-like score (clamped between 0 and 1)
            try:
                score = max(0.0, 1.0 - float(distance))
            except Exception:
                score = None

        # Try to find the resume id in SQLite by matching parsed_data.email
        resume_id = None
        try:
            # naive scan (acceptable for small datasets)
            candidates = db.query(Resume).all()
            for r in candidates:
                try:
                    if r.parsed_data and isinstance(r.parsed_data, dict) and r.parsed_data.get("email") == email:
                        resume_id = r.id
                        break
                except Exception:
                    continue
        except Exception:
            resume_id = None

        matches.append({
            "email": email,
            "resume_id": resume_id,
            "score": score,
            "best_section": info.get("section"),
            "document_preview": (info.get("document")[:200] + "...") if info.get("document") and len(info.get("document")) > 200 else info.get("document"),
        })

    # sort by score desc (None scores go last)
    matches_sorted = sorted(matches, key=lambda x: (x["score"] is not None, x["score"] if x["score"] is not None else -1), reverse=True)

    return {"status": "ok", "query": job_description[:200], "matches": matches_sorted[:top_k]}
