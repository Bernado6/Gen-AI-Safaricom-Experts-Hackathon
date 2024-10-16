from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import logging
import random
import time
import hashlib
from fastapi.background import BackgroundTasks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model
try:
    model = joblib.load('model/svc_model.pkl')
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Could not load the model. Ensure 'svc_model.pkl' is available in the 'model' directory.")

# Create the FastAPI app
app = FastAPI(
    title="Iris Flower Classification API",
    description="A production-ready API that classifies iris flowers based on sepal and petal measurements.",
    version="1.0.0",
    contact={
        "name": "Samurai",
        "email": "samurai@example.com",
        "url": "https://www.example.com/samurai",
    },
)

# Define the request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchIrisFeatures(BaseModel):
    features: List[IrisFeatures]

# Unique identifier for the code
CODE_SIGNATURE = hashlib.md5("IrisAPI_v1.0.0_Production".encode()).hexdigest()

# Dependency for input validation
def validate_input(features: IrisFeatures):
    if not (0 < features.sepal_length < 10 and
            0 < features.sepal_width < 10 and
            0 < features.petal_length < 10 and
            0 < features.petal_width < 10):
        raise HTTPException(status_code=400, detail="Invalid input values. All measurements should be between 0 and 10 cm.")
    return features

# Background task for logging predictions
def log_prediction(prediction: str, features: dict):
    logger.info(f"Prediction: {prediction}, Features: {features}")

@app.post("/predict", summary="Predict Iris Species", tags=["Prediction"])
async def predict(features: IrisFeatures = Depends(validate_input), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        # Prepare the data for prediction
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Perform the prediction
        prediction = model.predict(input_data)
        prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        result = prediction_map[int(prediction[0])]
        
        # Log the prediction asynchronously
        background_tasks.add_task(log_prediction, result, features.dict())
        
        # Return the response with meaningful prediction
        return {
            "prediction": result,
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", summary="Batch Predict Iris Species", tags=["Prediction"])
async def predict_batch(batch_features: BatchIrisFeatures):
    try:
        # Validate input
        for features in batch_features.features:
            validate_input(features)
        
        # Prepare the data for batch prediction
        input_data = np.array([
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ] for features in batch_features.features
        ])
        
        # Perform the prediction
        predictions = model.predict(input_data)
        prediction_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        # Return the response with meaningful predictions
        results = [prediction_map[int(pred)] for pred in predictions]
        logger.info(f"Batch prediction completed. Results: {results}")
        return {
            "predictions": results,
            "code_signature": CODE_SIGNATURE
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/predict_random", summary="Get Random Iris Prediction", tags=["Prediction"])
async def predict_random():
    try:
        # Generate random features within typical ranges for iris flowers
        random_features = IrisFeatures(
            sepal_length=random.uniform(4.0, 8.0),
            sepal_width=random.uniform(2.0, 4.5),
            petal_length=random.uniform(1.0, 7.0),
            petal_width=random.uniform(0.1, 2.5)
        )
        
        # Log generated random features
        logger.info(f"Generated random features: {random_features}")
        
        # Use the predict endpoint for consistency
        result = await predict(random_features)
        
        # Return the response with random features and prediction
        return {
            "random_features": random_features.dict(),
            "prediction": result["prediction"],
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logger.error(f"Random prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Random prediction failed: {str(e)}")

@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check():
    return {"status": "Healthy", "code_signature": CODE_SIGNATURE}

@app.get("/model_info", summary="Get Model Information", tags=["Info"])
async def model_info():
    return {
        "model": "Support Vector Classifier (SVC)",
        "version": "1.0",
        "description": "A model trained on the Iris dataset to classify iris flower species.",
        "code_signature": CODE_SIGNATURE
    }

@app.get("/simulate_workload", summary="Simulate Workload", tags=["Testing"])
async def simulate_workload(seconds: Optional[int] = 1):
    try:
        # Log the workload simulation request
        logger.info(f"Simulating workload for {seconds} seconds")
        
        # Simulate some workload
        time.sleep(seconds)
        
        # Return a success response
        return {
            "message": f"Successfully simulated workload for {seconds} seconds",
            "code_signature": CODE_SIGNATURE
        }
    except Exception as e:
        logger.error(f"Workload simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workload simulation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)