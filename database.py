from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Updated connection string for SQLAlchemy
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:Rahul@123@127.0.0.1:5432/BCGDB"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()