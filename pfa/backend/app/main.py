import os
from pathlib import Path
from typing import List, Dict
import asyncio
from io import BytesIO

import PyPDF2
import pandas as pd
from anthropic import Anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from uvicorn import run
import logging
from backend.app.config import project_dir
from backend.app.utils import get_logger

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

class AccountSummary(BaseModel):
    total_assets: float
    total_credit: float
    net_worth: float
    monthly_summary: Dict[str, Dict[str, float]]  # Format: {month: {category: amount}}

class FinancialAnalyzer:
    def __init__(self):
        self._transactions: List[Transaction] = []

    def add_transactions(self, new_transactions: List[Transaction]):
        self._transactions.extend(new_transactions)

    def get_summary(self) -> AccountSummary:
        # Group transactions by month
        monthly_transactions = {}
        total_assets = 0
        total_credit = 0

        for trans in self._transactions:
            # Convert date string to month key (e.g., "2024-03")
            month = trans.date[:7]
            
            if month not in monthly_transactions:
                monthly_transactions[month] = {}
            
            if trans.category not in monthly_transactions[month]:
                monthly_transactions[month][trans.category] = 0
                
            monthly_transactions[month][trans.category] += trans.amount
            
            if trans.type == 'income':
                total_assets += trans.amount
            else:  # expense
                total_credit += trans.amount

        return AccountSummary(
            total_assets=total_assets,
            total_credit=total_credit,
            net_worth=total_assets + total_credit,  # credit is negative
            monthly_summary=monthly_transactions
        )

# Create a global analyzer instance
financial_analyzer = FinancialAnalyzer()

class FinancialParser:
    def __init__(self):
        pass

    async def parse_td_pdf(self, fp: str) -> \
    List[Transaction]:
        # Convert PDF to text
        reader = PyPDF2.PdfReader(fp)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Use Claude to parse transactions
        prompt = f"""Extract all financial transactions from this bank statement into a structured format.
        For each transaction, identify:
        1. Date (format should be MM/DD/YYYY)
        2. Description
        3. Amount (calculated based on withdrawals and deposits per record, specifically is deposits subtracts withdrawals)
        4. Category (based on description per record)

        Bank statement text:
        {text}

        Return the data in this format: 
        Date|Description|Amount|Category"""

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

    async def parse_td_credit_csv(self, fp: str) -> List[Transaction]:
        df = pd.read_csv(fp)

        # Use Claude to parse transactions
        prompt = f"""Extract all financial transactions from the given dataframe into a structured format.
        For each transaction, identify:
        1. Date (format should be MM/DD/YYYY)
        2. Description
        3. Amount (usually column 2 is expense and column 3 is income, amount should be income subtracts expense)
        4. Category (based on description per record)

        Reference dataframe:
        {df}

        Return the data in this format:
        Date|Description|Amount|Category"""

        message = anthropic.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        categories = message.content[0].text.strip().split('\n')

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
                    source='td_credit'
                ))

        return transactions


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
    source_types: List[str] = Form(description="Source types for each file (td_chequing or td_credit)")
):
    logger.debug(f"Files: {[f.filename for f in files]}")
    logger.debug(f"Raw source types: {source_types}")
    
    # Ensure source_types is a list of individual types
    if len(source_types) == 1 and ',' in source_types[0]:
        source_types = source_types[0].split(',')
    
    logger.debug(f"Processed source types: {source_types}")
    logger.debug(f"Length of files: {len(files)}")
    logger.debug(f"Length of source types: {len(source_types)}")
    
    if len(files) != len(source_types):
        raise HTTPException(status_code=400, detail="Number of files must match number of source types")
        
    all_transactions = []
    
    for file, source_type in zip(files, source_types):
        content = await file.read()
        temp_file = Path(project_dir, "data/temp", file.filename)
        
        # Ensure temp directory exists
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write uploaded file to temp location
            with open(temp_file, "wb") as f:
                f.write(content)
                
            # Parse based on source type
            if source_type == "td_chequing":
                transactions = await parser.parse_td_pdf(str(temp_file))
            elif source_type == "td_credit":
                transactions = await parser.parse_td_credit_csv(str(temp_file))
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
                
            all_transactions.extend(transactions)
            
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
                
    # Add transactions to analyzer
    financial_analyzer.add_transactions(all_transactions)
    
    return {"transactions": all_transactions}

@app.get("/api/summary")
async def get_summary() -> AccountSummary:
    summary = financial_analyzer.get_summary()
    return summary


if __name__ == "__main__":
    # Run the app with uvicor
    async def test_upload_statements():
        print("Testing upload statements...")
        # Create test files list with their corresponding source types
        test_files = [
            (Path(project_dir, "data/raw/TD_MINIMUM_CHEQUING_ACCOUNT_3250-6519612_Oct_31-Nov_29_2024.pdf"), "td_chequing"),
            (Path(project_dir, "data/raw/TD_CREDIT_Oct_16_Nov_15.csv"), "td_credit")
        ]
        
        # Create FastAPI UploadFile objects and source_types list
        files = []
        source_types = []
        
        for file_path, source_type in test_files:
            # Open file and create UploadFile
            with open(file_path, "rb") as f:
                file_content = f.read()
                upload_file = UploadFile(
                    filename=file_path.name,
                    file=BytesIO(file_content)
                )
                files.append(upload_file)
                source_types.append(source_type)
        
        # Call the endpoint with proper parameters
        result = await upload_statements(files=files, source_types=source_types)
        print(f"Uploaded {len(result['transactions'])} transactions")

    async def test_get_summary():
        print("Testing get summary...")
        summary = await get_summary()
        print(f"Summary: {summary}")

    # Run async function for testing
    asyncio.run(test_upload_statements())
    asyncio.run(test_get_summary())

    # Start the FastAPI server
    run(app, host="127.0.0.1", port=8000)
