# Personal Financial Agent (PFA)

A comprehensive financial tracking and analysis tool that integrates:
- Banking accounts
- Investment portfolios
- Payment platforms (WeChat/Alipay)
- Real-time market data
- AI-powered insights

## Features

- Portfolio tracking with real-time updates
- Asset allocation visualization
- Performance analytics
- Dividend tracking
- AI-powered investment suggestions
- Multi-currency support
- Bank account aggregation
- Transaction history and categorization

## Tech Stack

Frontend: SwiftUI (✅ Good choice)
Pros:
- Native iOS performance
- Modern declarative syntax
- Great for financial data visualization
- Built-in security features

Backend: Suggest simplifying to just Python
Reasoning:
- Python excels at:
  - Data processing/analysis
  - LLM integration (OpenAI, Anthropic APIs)
  - PDF/document parsing
  - Financial calculations
- Adding Java would increase complexity without clear benefits
- FastAPI framework provides high performance

Database: Recommend PostgreSQL
- Strong support for financial data
- ACID compliance
- JSON support for flexible statement formats

## Test

### Backend Testing

Unit Tests (pytest):
   - Statement parsing logic
   - API endpoints
   - Database operations

Integration Tests:
   - API flow testing
   - Database interactions
   - LLM integration

### Frontend Testing

Unit Tests (XCTest):
   - ViewModels
   - Data processing
   - UI Components

UI Tests:
   - Navigation flow
   - User interactions
   - Data presentation

End-to-End Testing:
   - Complete user flows
   - Real device testing
   - Network conditions



cursor使用感受：新任务正确率很高，token长了以后后面容易越改越乱；最新3.5和hauki同样prompt差距蛮大的

codebase太牛了，帮我scan all codes and debug example说一下