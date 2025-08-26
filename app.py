# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from AnomalyDetectionIsolationForest import LogAnomalyDetector

app = FastAPI(title="Log Anomaly Detection API", version="1.0.0")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Array of JSON objects (each = one row).")
    selected_features: List[str] = Field(..., description="Columns to use for anomaly detection.")
    contamination: Optional[float] = Field(0.1, ge=0.01, le=0.5, description="Proportion of outliers (0.01–0.5).")
    random_state: Optional[int] = 42

class PredictResponse(BaseModel):
    n_rows: int
    n_features_used: int
    selected_features: List[str]
    contamination: float
    predictions: List[int]          # 1 = normal, -1 = anomaly
    anomaly_scores: List[float]     # lower = more anomalous
    n_anomalies: int
    n_normal: int

@app.get("/health")
def health():
    return {"status": "ok"}

# Can use in absance when need not to upload .csv file alternatively.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Convert JSON records to DataFrame
    if not req.records:
        raise HTTPException(status_code=400, detail="`records` cannot be empty.")
    df = pd.DataFrame(req.records)

    # Validate selected features exist
    missing = [c for c in req.selected_features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in data: {missing}")

    # Run detector
    try:
        detector = LogAnomalyDetector(contamination=req.contamination, random_state=req.random_state)
        predictions, scores = detector.fit_predict(df, req.selected_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # Summaries
    n_anom = int((pd.Series(predictions) == -1).sum())
    n_norm = int((pd.Series(predictions) == 1).sum())

    return PredictResponse(
        n_rows=len(df),
        n_features_used=len(req.selected_features),
        selected_features=req.selected_features,
        contamination=req.contamination,
        predictions=list(map(int, predictions)),
        anomaly_scores=list(map(float, scores)),
        n_anomalies=n_anom,
        n_normal=n_norm,
    )

# add to app.py
from fastapi import UploadFile, File, Query
from io import BytesIO

@app.post("/predict-csv", response_model=PredictResponse)
async def predict_csv(
    file: UploadFile = File(..., description="CSV file"),
    selected_features: List[str] = Query(..., description="Repeat ?selected_features=col for each feature"),
    contamination: float = Query(0.1, ge=0.01, le=0.5),
    random_state: int = 42
):
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in CSV: {missing}")

    try:
        #detector = LogAnomalyDetector(contamination=contamination, random_state=random_state)
        print('About to call IsolationForest ------>')
        detector = LogAnomalyDetector(contamination=contamination)
        # Get the file path from the uploaded file
        file_path = f"node_{node_id}_data.csv"  # Or your preferred filename
        file_location = os.path.join('data', file_path)
        print('file_location is =====++++++> ', file_location)
        # Save the uploaded file first
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        print('file uploaded is =====++++++> ', file)
        # Call fit_predict with the file path
        predictions, scores = detector.fit_predict(file_location, selected_features)
        #predictions, scores = detector.fit_predict(df, selected_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    n_anom = int((pd.Series(predictions) == -1).sum())
    n_norm = int((pd.Series(predictions) == 1).sum())

    return PredictResponse(
        n_rows=len(df),
        n_features_used=len(selected_features),
        selected_features=selected_features,
        contamination=contamination,
        predictions=list(map(int, predictions)),
        anomaly_scores=list(map(float, scores)),
        n_anomalies=n_anom,
        n_normal=n_norm,
    )