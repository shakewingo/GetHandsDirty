import os
import re
from pathlib import Path
from typing import List, Dict
import asyncio
from io import BytesIO
from datetime import datetime, timezone

import PyPDF2
import yfinance as yf
import pandas as pd
from anthropic import Anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from uvicorn import run
import logging
from sqlalchemy.orm import Session

from backend.app.config import project_dir
from backend.app.utils import get_logger
from backend.app.models import TransactionModel, AssetModel, CreditModel
from backend.app.database import get_db, engine

logger = get_logger(__file__, level=logging.DEBUG)
anthropic = Anthropic()
model="claude-3-5-sonnet-20241022"

class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    category: str
    type: str
    source: str

class Asset(BaseModel):
    id: int = 0
    asset_type: str
    market_value: float | None = None
    market_share: float | None = None
    currency: str
    created_at: str = ""

class Credit(BaseModel):
    id: int = 0
    credit_type: str
    market_value: float  # Required field
    currency: str
    created_at: str = ""

class AccountSummary(BaseModel):
    total_assets: float
    total_credit: float
    net_worth: float
    monthly_summary: Dict[str, Dict[str, float]]  # Format: {month: {category: amount}}

class FinancialAnalyzer:
    def add_transactions(self, new_transactions: List[Transaction], db: Session):
        for trans in new_transactions:
            db_transaction = TransactionModel(
                date=trans.date,
                description=trans.description,
                amount=trans.amount,
                category=trans.category,
                type=trans.type,
                source=trans.source
            )
            db.add(db_transaction)
        db.commit()

    def add_asset(self, asset: Asset, db: Session):
        if asset.market_value is None and asset.market_share is None:
            raise HTTPException(
                status_code=400, 
                detail="Either market_value or market_share must be provided"
            )
            
        # Calculate market value if only share is provided
        if asset.market_share is not None:
            market_value = self.calculate_market_value(
                asset.asset_type, 
                asset.market_share
            )
            if market_value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not calculate market value for {asset.asset_type}. Please verify the ticker symbol."
                )
            asset.market_value = market_value
            logger.info(f"Calculated market value: {market_value} for {asset.asset_type}")

        db_asset = AssetModel(
            asset_type=asset.asset_type,
            market_value=asset.market_value,
            market_share=asset.market_share,
            currency=asset.currency,
            created_at=datetime.strptime(asset.created_at, "%Y-%m-%d %H:%M:%S") 
                if asset.created_at else datetime.now(timezone.utc)
        )
        db.add(db_asset)
        db.commit()
        db.refresh(db_asset)  # Refresh to get the updated data
        logger.info(f"Added asset to DB: {db_asset.__dict__}")

    def add_credit(self, credit: Credit, db: Session):
        db_credit = CreditModel(
            credit_type=credit.credit_type,
            market_value=credit.market_value,
            currency=credit.currency,
            created_at=datetime.strptime(credit.created_at, "%Y-%m-%d %H:%M:%S") if credit.created_at else datetime.now(timezone.utc)
        )
        db.add(db_credit)
        db.commit()
        
    # avoid calling everytime when refresh/edit/add asset/credit, too slow
    def get_summary(self, db: Session) -> AccountSummary:
        # Group transactions by month
        monthly_transactions = {}
        
        for trans in db.query(TransactionModel).all():
            # Convert date string to month key (e.g., "2024-03")
            year = trans.date[-4:]
            month = trans.date[:2]
            month_key = f"{year}-{month}"
            if month not in monthly_transactions:
                monthly_transactions[month_key] = {}
            
            if trans.category not in monthly_transactions[month_key]:
                monthly_transactions[month_key][trans.category] = 0
                
            monthly_transactions[month_key][trans.category] += trans.amount

        # Get all assets and credits
        assets = [
            Asset(
                asset_type=asset.asset_type,
                market_value=self.calculate_market_value(asset.asset_type, asset.market_share) if asset.market_share else asset.market_value,
                market_share=asset.market_share,
                currency=asset.currency,
                created_at=asset.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
            for asset in db.query(AssetModel).all()
        ]

        credits = [
            Credit(
                credit_type=credit.credit_type,
                market_value=credit.market_value,
                currency=credit.currency,
                created_at=credit.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
            for credit in db.query(CreditModel).all()
        ]

        assets_results = [f"{asset.asset_type} {asset.currency} {asset.market_value}" for asset in assets]
        credits_results = [f"{credit.credit_type} {credit.currency} {credit.market_value}" for credit in credits]
        
        # Use Claude to calculate totals (keep existing prompts)
        prompt1 = f"""
        Analyzing {assets_results} which represent its type, currency and value in the string:
        1) Based on currency e.g USD/EUR/RMB/CAD etc, converting the value to RMB based on the realtime exchange rate.
        2) Sum the converted value, and ensure your answer include the extract words: "Total sum in RMB = XXXX RMB" which XXXX is a number
        If list {assets_results} is empty, "Total sum in RMB = 0 RMB"
        """
        message1 = anthropic.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt1
            }]
        )

        prompt2 = f"""
        Analyzing {credits_results} which represent its type, currency and value in the string:
        1) Based on currency e.g USD/EUR/RMB/CAD etc, converting the value to RMB based on the realtime exchange rate.
        2) Sum the converted value, and ensure your answer include the extract words: "Total sum in RMB = XXXX RMB" which XXXX is a number
        If list {credits_results} is empty, "Total sum in RMB = 0 RMB"
        """
        message2 = anthropic.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt2
            }]
        )
        logger.debug('messag1: %s', message1.content[0].text)
        assets_value = message1.content[0].text.split('Total sum in RMB = ')[1].split(' RMB')[0]
        logger.debug(f"assets_value: {assets_value}")
        total_assets = float(assets_value.replace(',', ''))

        logger.info('messag2: %s', message2.content[0].text)
        credits_value = message2.content[0].text.split('Total sum in RMB = ')[1].split(' RMB')[0]
        logger.debug(f"credits_value: {credits_value}")
        total_credit = float(credits_value.replace(',', ''))
        logger.debug(f"total_assets: {total_assets}, total_credit: {total_credit}")


        return AccountSummary(
            total_assets=round(total_assets, 2),
            total_credit=round(total_credit, 2),
            net_worth=round(total_assets + total_credit, 2),
            monthly_summary=monthly_transactions
        )
    
    @staticmethod
    def calculate_market_value(asset_type: str, market_share: float) -> float | None:
        """Calculate market value based on market share"""
        return 0
        # try:
        #     ticker = yf.Ticker(asset_type)
        #     price = ticker.info.get('currentPrice')
        #     if price is None:
        #         price = ticker.info.get('regularMarketPrice')  # Try alternative price field
            
        #     logger.info(f'asset_type: {asset_type}, price: {price}, market_share: {market_share}')
            
        #     if price is None:
        #         logger.error(f"Could not get price for {asset_type}, return 0")
        #         return None
            
        #     return price * market_share 
        # except Exception as e:
        #     logger.error(f"Error calculating market value for {asset_type}: {e}, return 0")
        #     return None

    def get_assets(self, db: Session) -> List[Asset]:
        assets = db.query(AssetModel).all()
        result = []
        
        for asset in assets:
            # Calculate current market value if market_share exists
            if asset.market_share is not None:
                current_value = self.calculate_market_value(
                    asset.asset_type,
                    asset.market_share
                )
                # Update market value in DB
                asset.market_value = current_value
                
            result.append(Asset(
                id=asset.id,
                asset_type=asset.asset_type,
                market_value=asset.market_value,
                market_share=asset.market_share,
                currency=asset.currency,
                created_at=asset.created_at.strftime("%Y-%m-%d %H:%M:%S") if asset.created_at else ""
            ))
        
        # Commit updates to DB
        db.commit()
        return result

    def get_asset_details(self, asset_type: str, currency: str, db: Session) -> List[Asset]:
        assets = db.query(AssetModel).filter(
            AssetModel.asset_type == asset_type,
            AssetModel.currency == currency
        ).all()
        
        result = []
        for asset in assets:
            # Calculate current market value if market_share exists
            if asset.market_share is not None:
                current_value = self.calculate_market_value(
                    asset.asset_type,
                    asset.market_share
                )
                # Update market value in DB
                asset.market_value = current_value
                
            result.append(Asset(
                id=asset.id,
                asset_type=asset.asset_type,
                market_value=asset.market_value,
                market_share=asset.market_share,
                currency=asset.currency,
                created_at=asset.created_at.strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        # Commit updates to DB
        db.commit()
        return result

    def get_credits(self, db: Session) -> List[Credit]:
        return [
            Credit(
                id=credit.id,
                credit_type=credit.credit_type,
                market_value=credit.market_value,
                currency=credit.currency,
                created_at=credit.created_at.strftime("%Y-%m-%d %H:%M:%S") if credit.created_at else ""
            )
            for credit in db.query(CreditModel).all()
        ]

    def get_credit_details(self, credit_type: str, currency: str, db: Session) -> List[Credit]:
        return [
            Credit(
                id=credit.id,
                credit_type=credit.credit_type,
                market_value=credit.market_value,
                currency=credit.currency,
                created_at=credit.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
            for credit in db.query(CreditModel).filter(
                CreditModel.credit_type == credit_type,
                CreditModel.currency == currency
            ).all()
        ]

    def get_grouped_assets(self, db: Session) -> List[Asset]:
        # Get all assets
        assets = db.query(AssetModel).all()
        logger.debug(f"Raw assets from DB: {assets}")
        
        # Update market values first
        for asset in assets:
            if asset.market_share is not None:
                current_value = self.calculate_market_value(
                    asset.asset_type,
                    asset.market_share
                )
                asset.market_value = current_value
        
        db.commit()
        
        # Create a dictionary to group assets
        grouped = {}
        for asset in assets:
            key = (asset.asset_type, asset.currency)
            if key not in grouped:
                grouped[key] = {
                    'id': asset.id,
                    'asset_type': asset.asset_type,
                    'currency': asset.currency,
                    'market_value': 0,
                    'market_share': asset.market_share,
                    'created_at': asset.created_at
                }
            grouped[key]['market_value'] += asset.market_value
        
        # Convert grouped dictionary to list of Asset objects
        result = [
            Asset(
                id=data['id'],
                asset_type=data['asset_type'],
                market_value=data['market_value'],
                market_share=data['market_share'],
                currency=data['currency'],
                created_at=data['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            )
            for data in grouped.values()
        ]
        
        logger.debug(f"Final result: {result}")
        return result

    def get_grouped_credits(self, db: Session) -> List[Credit]:
        # Get all credits
        credits = db.query(CreditModel).all()
        
        # Create a dictionary to group credits
        grouped = {}
        for credit in credits:
            key = (credit.credit_type, credit.currency)
            if key not in grouped:
                grouped[key] = {
                    'id': credit.id,
                    'credit_type': credit.credit_type,
                    'currency': credit.currency,
                    'market_value': 0,
                    'created_at': credit.created_at
                }
            grouped[key]['market_value'] += credit.market_value

        # Convert grouped dictionary to list of Credit objects
        return [
            Credit(
                id=data['id'],
                credit_type=data['credit_type'],
                market_value=data['market_value'],
                currency=data['currency'],
                created_at=data['created_at'].strftime("%Y-%m-%d %H:%M:%S")
            )
            for data in grouped.values()
        ]

    # TODO: maybe use later
    def update_market_values(self, db: Session):
        """Update market values for assets with only market share"""
        assets = db.query(AssetModel).filter(
            AssetModel.market_share.isnot(None),
            AssetModel.market_value.is_(None)
        ).all()

        for asset in assets:
            market_value = self.calculate_market_value(
                asset.asset_type,
                asset.market_share
            )
            if market_value is not None:
                asset.market_value = market_value
        
        db.commit()

# Create a global analyzer instance
financial_analyzer = FinancialAnalyzer()

class FinancialParser:
    def __init__(self):
        pass

    def _check_file_type(self, file_path: str) -> str:
        """Check if file is PDF or CSV based on extension"""
        file_extension = file_path.lower().split('.')[-1]
        if file_extension not in ['pdf', 'csv']:
            raise ValueError(f"Unsupported file type: {file_extension}. Only PDF and CSV files are supported.")
        return file_extension

    async def parse_statement(self, fp: str) -> List[Transaction]:
        # Check file type first
        file_type = self._check_file_type(fp)
        if file_type == 'pdf':
            # Convert PDF to text
            reader = PyPDF2.PdfReader(fp)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        else:
            # Read CSV file
            text = pd.read_csv(fp)

        # Use Claude to parse transactions
        prompt = f"""Extract all financial transactions from this bank statement into a structured format.
        For each transaction, identify:
        1. Date 
        2. Description
        3. Amount 
        4. Category

        Bank statement text:
        {text}
 
        Carefually do the following: 
        1) For date, convert format to be MM/DD/YYYY 
        2) For amount, the calculation depends on source_types. Specifically,
        for TD_CHEQUING, calculated based on deposits subtracts withdrawals
        for TD_CREDIT, colunm 1 to 5 represents date, description, expense, income, balance separately, so amount is calculate based on income subtracts expense, where income is zero if it's empty
        for CMB_CHEQUING, amount is equal to transaction amount.
        After get the number, always convert it to RMB based on the realtime exchange rate and then keep only the float as the final "amount"
        3) For category, based on description or counter party
 
        Return Date|Description|Amount|Category"""

        message = anthropic.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        transactions = []
        for line in message.content[0].text.split('\n'):
            if '|' in line:
                date, desc, amount, category = line.split('|')
                logger.debug(f"Parsed transaction: {date}, {desc}, {amount}, {category}")
                transactions.append(Transaction(
                    date=date.strip(),
                    description=desc.strip(),
                    amount=float(amount),
                    category=category.strip(),
                    type='income' if float(amount) > 0 else 'expense',
                    source='td_chequing'
                ))

        return transactions

import os
port = int(os.environ.get("PORT", 8000))
# Listen on all interfaces (0.0.0.0) to allow connections from other devices
host = os.environ.get("HOST", "0.0.0.0")
# host = os.environ.get("HOST", "127.0.0.1")  # This only allows connections from the same machine

app = FastAPI(title="Financial Tracker API")
parser = FinancialParser()

@app.get("/")
async def root():
    return {
        "message": "Financial Assistant API",
        "version": "1.0",
        "endpoints": {
            "/api/summary": "Get summary",
            "/docs": "API documentation"
        }
    }

@app.post("/api/upload_statements")
async def upload_statements(
    files: List[UploadFile] = File(...),
    source_types: List[str] = Form(description="Source types for each file (td_chequing, td_credit, cmb_chequing, cmb_credit)"),
    db: Session = Depends(get_db)
):
    logger.debug(f"Files: {[f.filename for f in files]}")
    logger.debug(f"Raw source types: {source_types}")
    
    if len(source_types) == 1 and ',' in source_types[0]:
        source_types = source_types[0].split(',')
    
    logger.debug(f"Processed source types: {source_types}")
    logger.debug(f"Length of files: {len(files)}")
    logger.debug(f"Length of source types: {len(source_types)}")
    
    valid_source_types = ["td_chequing", "td_credit", "cmb_chequing", "cmb_credit"]
    
    if len(files) != len(source_types):
        raise HTTPException(status_code=400, detail="Number of files must match number of source types")
    
    for source_type in source_types:
        if source_type not in valid_source_types:
            raise HTTPException(status_code=400, detail=f"Invalid source type: {source_type}. Must be one of {valid_source_types}")
        
    all_transactions = []
    
    for file, source_type in zip(files, source_types):
        content = await file.read()
        temp_file = Path(project_dir, "data/temp", file.filename)
        
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(temp_file, "wb") as f:
                f.write(content)
                
            if source_type in valid_source_types:
                transactions = await parser.parse_statement(str(temp_file))
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
                
            all_transactions.extend(transactions)
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    financial_analyzer.add_transactions(all_transactions, db)
    
    return {"transactions": all_transactions}

@app.get("/api/summary")
async def get_summary(db: Session = Depends(get_db)) -> AccountSummary:
    return financial_analyzer.get_summary(db)

@app.post("/api/assets")
async def add_asset(asset: Asset, db: Session = Depends(get_db)):
    logger.debug(f"Received asset to add: {asset}")
    if not asset.created_at:
        asset.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if asset.market_value and asset.market_value <= 0:
        raise HTTPException(status_code=400, detail="Asset market value must be positive")
    financial_analyzer.add_asset(asset, db)
    return {"status": "success"}

@app.get("/api/assets")
async def get_assets(db: Session = Depends(get_db)) -> List[Asset]:
    return financial_analyzer.get_assets(db)

@app.post("/api/credits")
async def add_credit(credit: Credit, db: Session = Depends(get_db)):
    if not credit.created_at:
        credit.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if credit.market_value >= 0:
        raise HTTPException(status_code=400, detail="Credit market value must be negative")
    financial_analyzer.add_credit(credit, db)
    return credit

@app.get("/api/credits")
async def get_credits(db: Session = Depends(get_db)) -> List[Credit]:
    return financial_analyzer.get_credits(db)

@app.get("/api/asset_details")
async def get_asset_details(
    asset_type: str,
    currency: str,
    db: Session = Depends(get_db)
) -> List[Asset]:
    return financial_analyzer.get_asset_details(asset_type, currency, db)

@app.get("/api/credit_details")
async def get_credit_details(
    credit_type: str,
    currency: str,
    db: Session = Depends(get_db)
) -> List[Credit]:
    return financial_analyzer.get_credit_details(credit_type, currency, db)

@app.get("/api/grouped_assets")
async def get_grouped_assets(db: Session = Depends(get_db)) -> List[Asset]:
    logger.debug("Entering /api/grouped_assets endpoint")
    result = financial_analyzer.get_grouped_assets(db)
    logger.debug(f"Endpoint result: {result}")
    return result

@app.get("/api/grouped_credits")
async def get_grouped_credits(db: Session = Depends(get_db)) -> List[Credit]:
    return financial_analyzer.get_grouped_credits(db)

@app.put("/api/assets/{asset_id}")
async def update_asset(asset_id: int, asset: Asset, db: Session = Depends(get_db)):
    logger.debug(f"Updating asset with id: {asset_id}")
    logger.debug(f"Update data: {asset}")
    
    db_asset = db.query(AssetModel).filter(AssetModel.id == asset_id).first()
    logger.debug(f"Found asset in DB: {db_asset}")
    
    if not db_asset:
        logger.error(f"Asset not found with id: {asset_id}")
        raise HTTPException(status_code=404, detail="Asset not found")
    
    db_asset.market_value = asset.market_value
    db_asset.market_share = asset.market_share
    db_asset.currency = asset.currency
    try:
        db.commit()
        logger.debug(f"Successfully updated asset: {db_asset}")
        return asset
    except Exception as e:
        logger.error(f"Error updating asset: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/assets/{asset_id}")
async def delete_asset(asset_id: int, db: Session = Depends(get_db)):
    logger.debug(f"Deleting asset with id: {asset_id}")
    
    db_asset = db.query(AssetModel).filter(AssetModel.id == asset_id).first()
    logger.debug(f"Found asset in DB: {db_asset}")
    
    if not db_asset:
        logger.error(f"Asset not found with id: {asset_id}")
        raise HTTPException(status_code=404, detail="Asset not found")
    
    try:
        db.delete(db_asset)
        db.commit()
        logger.debug(f"Successfully deleted asset with id: {asset_id}")
        return {"message": "Asset deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting asset: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/credits/{credit_id}")
async def update_credit(credit_id: int, credit: Credit, db: Session = Depends(get_db)):
    logger.debug(f"Updating credit with id: {credit_id}")
    
    db_credit = db.query(CreditModel).filter(CreditModel.id == credit_id).first()
    if not db_credit:
        raise HTTPException(status_code=404, detail="Credit not found")
    
    if credit.market_value >= 0:
        raise HTTPException(status_code=400, detail="Credit market value must be negative")
    
    db_credit.market_value = credit.market_value
    db_credit.currency = credit.currency
    try:
        db.commit()
        return credit
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/credits/{credit_id}")
async def delete_credit(credit_id: int, db: Session = Depends(get_db)):
    logger.debug(f"Deleting credit with id: {credit_id}")
    
    db_credit = db.query(CreditModel).filter(CreditModel.id == credit_id).first()
    logger.debug(f"Found credit in DB: {db_credit}")
    
    if not db_credit:
        logger.error(f"Credit not found with id: {credit_id}")
        raise HTTPException(status_code=404, detail="Credit not found")
    
    try:
        db.delete(db_credit)
        db.commit()
        logger.debug(f"Successfully deleted credit with id: {credit_id}")
        return {"message": "Credit deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting credit: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    run(app, host=host, port=port)
