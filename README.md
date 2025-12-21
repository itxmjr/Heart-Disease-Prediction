# Heart Disease Prediction System

A machine learning project to predict heart disease risk based on patient health data.

## Features

- **Data Analysis**: Comprehensive Exploratory Data Analysis (EDA) with visualizations.
- **Machine Learning**: Compare Logistic Regression, Decision Tree, and Random Forest models.
- **Web Application**: Interactive Streamlit app for real-time predictions.
- **Interpretability**: Feature importance and risk factor analysis.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Kaggle API Setup**
   Ensure you have your Kaggle API credentials set up to download the dataset automatically.
   - Place `kaggle.json` in `~/.kaggle/`

## Usage

### 1. Run the Full Pipeline
To download data, clean it, run EDA, and train models:
```bash
python main.py
```
This will:
- Download the dataset to `data/`
- Generate analysis plots in `outputs/`
- Save the trained model to `models/heart_model.pkl`

### 2. Launch the Web App
Start the interactive dashboard:
```bash
streamlit run app.py
```

## Project Structure

- `src/`: Source code modules
  - `dataloader.py`: Data downloading and loading
  - `preprocessor.py`: process missing values and duplicates
  - `eda.py`: Visualization functions
  - `model.py`: ML model definitions and training logic
- `app.py`: Streamlit web application
- `main.py`: Pipeline orchestration script
- `data/`: Dataset storage
- `outputs/`: Generated plots and metrics
- `models/`: Saved model artifacts

## Disclaimer
This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
