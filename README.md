# Heart Disease Prediction System

An AI-powered cardiovascular risk assessment tool using machine learning. This application features a Flask backend API and an interactive web frontend.

## Features

- **ML-Based Prediction**: Uses scikit-learn trained model to predict heart disease risk
- **Real-time Analysis**: Instant predictions based on clinical parameters
- **Risk Assessment**: Provides confidence levels and risk categorization
- **Responsive Design**: Works on desktop and mobile devices
- **Production Ready**: Deployed configuration for Render hosting

## Local Development

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd Heart-Disease-Predictor
```

2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Ensure model files are present
```
heart_disease_model.pkl
scaler.pkl
```

### Running Locally

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Deployment to Render

### Prerequisites
- GitHub account with your repository
- Render account (free tier available)

### Deployment Steps

1. **Push to GitHub**
   - Ensure all files including model files are committed
   - Push to your GitHub repository

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Sign up or log in
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Deployment**
   - Name: `heart-disease-predictor`
   - Environment: `Python 3`
   - Region: Select closest to you
   - Plan: Free (or upgrade as needed)
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically deploy from your GitHub repository
   - Your app will be available at: `https://your-app-name.onrender.com`

### Environment Variables (Optional)

Set these in Render dashboard if needed:
- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Automatically set by Render

## API Endpoints

### Health Check
```
GET /api/health
```
Returns server and model status.

### Prediction
```
POST /api/predict
Content-Type: application/json

{
  "Age": 45,
  "Sex": 1,
  "Chest_Pain_Type": 2,
  "Resting_Blood_Pressure": 140,
  "Cholesterol": 250,
  "Fasting_Blood_Sugar": 0,
  "Resting_ECG": 0,
  "Max_Heart_Rate": 150,
  "Exercise_Angina": 0,
  "ST_Depression": 1.5,
  "ST_Slope": 2
}
```

### Validation
```
POST /api/validate
Content-Type: application/json

{
  "Age": 45,
  "Sex": 1,
  "Chest_Pain_Type": 2,
  ...
}
```

## Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| Age | Integer | 1-120 | Patient age in years |
| Sex | Integer | 0-1 | 0=Female, 1=Male |
| Chest_Pain_Type | Integer | 1-4 | 1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic |
| Resting_Blood_Pressure | Float | 80-200 | Systolic BP in mm Hg |
| Cholesterol | Float | 100-600 | Serum cholesterol in mg/dl |
| Fasting_Blood_Sugar | Integer | 0-1 | 0=≤120 mg/dl, 1=>120 mg/dl |
| Resting_ECG | Integer | 0-2 | 0=Normal, 1=ST-T abnormality, 2=LVH |
| Max_Heart_Rate | Float | 60-220 | Maximum heart rate achieved |
| Exercise_Angina | Integer | 0-1 | 0=No, 1=Yes |
| ST_Depression | Float | 0-10 | ST depression induced by exercise |
| ST_Slope | Integer | 1-3 | 1=Upsloping, 2=Flat, 3=Downsloping |

## File Structure

```
Heart-Disease-Predictor/
├── app.py                      # Flask backend application
├── index.html                  # Frontend web interface
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── Procfile                    # Process file for web server
├── .gitignore                  # Git ignore rules
├── heart_disease_model.pkl     # Trained ML model
└── scaler.pkl                  # Feature scaler
```

## Notes

- The application displays a warning if model files are not found
- Predictions should be used for educational purposes only
- Always consult healthcare professionals for medical decisions
- The frontend automatically detects local vs production environment

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, please refer to the GitHub repository or contact the development team.
