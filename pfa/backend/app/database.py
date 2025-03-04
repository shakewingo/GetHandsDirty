from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pathlib import Path
import os
from .config import project_dir
from .models import Base, TransactionModel, AssetModel, CreditModel  # Import models

# Create database directory if it doesn't exist
db_dir = Path(project_dir, "data", "db")
db_dir.mkdir(parents=True, exist_ok=True)

# Database URL
db_file = Path(db_dir, "financial_tracker.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_file}"


# Create SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create all tables
Base.metadata.create_all(bind=engine)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 