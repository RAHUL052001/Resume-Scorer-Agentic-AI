from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import shutil
import os

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Save file to local storage (S3/Blob Layer in Diagram)
    upload_dir = "storage/resumes"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = f"{upload_dir}/{file.filename}"
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # 2. Save metadata to Postgres (Resume record store)
    new_resume = Resume(filename=file.filename, file_path=file_location)
    db.add(new_resume)
    db.commit()
    
    return {"info": f"file '{file.filename}' saved", "id": new_resume.id}


@app.post("/jd-intake/")
async def jd_intake(job_description: str):
    # Logic for JD intake module
    return {"status": "JD received", "length": len(job_description)}