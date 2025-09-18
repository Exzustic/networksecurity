# Network Security Project For Phishing Data
A Python project for phishind data analysis and prediction

## Table of Content
- [About](#about)
- [Technologies](#technologies)
- [Instalation](#instalation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Licesnce](#license)

## About
This project analyzes network data to detect phishing activities. It includes data preprocessing, model training and prediction functionalities.

## Technologies
- Python 3.12.9
- FastAPI (For API in `app.py`)
- Pandas, NumPy, Scikit-learn
- Docker

## Instalation 
1. Clone the repository:
```bash
git clone https://github.com/Exzustic/networksecurity
cd networksecurity
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv 
source venv\bin\activate # Linux/Mac
venv\Scripts\activate # Windows 
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables 
Create `.env` file in the root folder and required variables:
```env
MONGO_DB_URL = your_mongo_db_url
```
or you can choose not to do this

## Usage
- To run the API:
```bash
uvicorn app:app
```
- To run the main processing script:
```bash
python main.py
```

## Project Structure
```
networksecurity/
|
├── src/                # Source code
├── data_schema/        # schema of dataset
├── final_model/        # Trained model
├── Network_Data/       # Raw data (works when you don't have .env)
├── valid_data/         # Validation dataset
├── prediction_output/  # Prediction result
├── templates/          # HTML templates for API
├── log/                # Logs
├── app.py              # FastAPI API entry point 
├── main.py             # Main processing script
├── requirements.txt
├── Dockerfile
```

## License 
MIT License  
Copyright (c) 2025 Krish Naik

