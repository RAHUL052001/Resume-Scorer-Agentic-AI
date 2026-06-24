from sqlalchemy import Column, DateTime, Integer, String, Text, JSON
from datetime import datetime
from database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    parsed_data = Column(JSON, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default="uploaded", nullable=False)
