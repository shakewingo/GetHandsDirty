from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class TransactionModel(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    description = Column(String)
    amount = Column(Float)
    category = Column(String)
    type = Column(String)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class AssetModel(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    asset_type = Column(String)
    market_value = Column(Float)
    currency = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class CreditModel(Base):
    __tablename__ = "credits"

    id = Column(Integer, primary_key=True, index=True)
    credit_type = Column(String)
    market_value = Column(Float)
    currency = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow) 