from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from .database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)  # For S3 or local blob storage
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # For tracking AI processing