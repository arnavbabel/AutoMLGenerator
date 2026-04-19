from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import io
import numpy as np
import json
import os
import httpx

app = FastAPI(title="AutoML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"status": "AutoML API is running"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    columns = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            columns[col] = "numeric"
        else:
            columns[col] = "categorical"
    
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": columns
    }

class TrainRequest(BaseModel):
    target: str
    features: list[str]
    model_type: str
    test_size: float = 0.2
    random_seed: int = 42

@app.post("/train")
async def train_model(file: UploadFile = File(...), config: str = ""):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    config_data = TrainRequest(**json.loads(config))

    X = df[config_data.features]
    y = df[config_data.target]

    X = pd.get_dummies(X, drop_first=True)

    if config_data.model_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config_data.test_size, random_state=config_data.random_seed)

    if config_data.model_type == "classification":
        model = LogisticRegression() if len(np.unique(y)) == 2 else RandomForestClassifier(n_estimators=50, max_depth=10, random_state=config_data.random_seed)
    elif config_data.model_type == "regression":
        model = LinearRegression()
    elif config_data.model_type == "regression_ridge":
        model = Ridge()
    elif config_data.model_type == "regression_dt":
        model = DecisionTreeRegressor()
    elif config_data.model_type == "regression_rf":
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=config_data.random_seed)
    elif config_data.model_type == "classification":
        model = LogisticRegression()
    elif config_data.model_type == "classification_knn":
        model = KNeighborsClassifier()
    elif config_data.model_type == "classification_dt":
        model = DecisionTreeClassifier()
    elif config_data.model_type == "classification_rf":
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=config_data.random_seed)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    model.fit(X_train, y_train)

    if hasattr(model, "feature_importances_"):
        importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):
        importance = dict(zip(X.columns, np.abs(model.coef_).flatten().tolist()))
    else:
        importance = {}

    y_pred = model.predict(X_test)

    if config_data.model_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        return {"accuracy": accuracy, "f1_score": f1, "feature_importance": importance}
    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {"r2_score": r2, "mae": mae, "mse": mse, "feature_importance": importance}
    
class InterpretRequest(BaseModel):
    task_type: str
    model_type: str
    target: str
    features: list[str]
    metrics: dict
    feature_importance: dict

@app.post("/interpret")
async def interpret_results(req: InterpretRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    top_features = sorted(req.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ', '.join([k for k, v in top_features])
    metrics_str = ', '.join([f"{k}: {v}" for k, v in req.metrics.items()])

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": f"You are a data science assistant. Interpret these ML results in 3-4 concise sentences:\n- Task: {req.task_type}\n- Model: {req.model_type}\n- Target: {req.target}\n- Features: {', '.join(req.features)}\n- Performance: {metrics_str}\n- Top features: {top_str}\nExplain what the results mean, if performance is good, and one actionable recommendation. Be direct."
                }]
            },
            timeout=30.0
        )
    
    data = response.json()
    if "content" not in data:
        raise HTTPException(status_code=500, detail=f"Anthropic API error: {data}")
    text = data["content"][0]["text"]
    return {"interpretation": text}
