# Iris Flower Classification API

## 1. Project Overview

### Purpose and Objectives

The Iris Flower Classification API is a machine learning-based application designed to classify iris flowers into three species: setosa, versicolor, and virginica. The primary objective of this project is to provide a user-friendly and efficient API for predicting the species of an iris flower based on its sepal and petal measurements.

### Target Users

The target users of this API include:

- Data scientists and machine learning engineers who want to integrate iris flower classification into their applications or workflows.
- Researchers and students studying machine learning and botanical classification.
- Developers building applications or services related to plant identification or classification.

### Key Features

- **Single Prediction Endpoint**: Accepts sepal and petal measurements as input and returns the predicted iris species.
- **Batch Prediction Endpoint**: Allows for batch prediction of multiple sets of sepal and petal measurements, returning a list of predicted iris species.
- **Random Prediction Endpoint**: Generates random sepal and petal measurements within typical ranges for iris flowers and returns the predicted iris species for the random input.
- **Health Check Endpoint**: Provides a simple health check for the application, returning a "Healthy" status.
- **Model Information Endpoint**: Returns information about the pre-trained model, including its type, version, and a brief description.
- **Workload Simulation Endpoint**: Simulates a workload for a specified number of seconds, useful for testing latency and performance.

## 2. Comprehensive Technical Architecture

### System Components

The Iris Flower Classification API consists of the following main components:

1. **FastAPI Application**: The core component that handles API requests and responses, and coordinates the data flow between the model and the API endpoints.
2. **Pre-trained SVC Model**: A Support Vector Classifier (SVC) model trained on the Iris dataset, used for making predictions based on sepal and petal measurements.
3. **Data Preprocessing**: A module responsible for preparing the input data in the required format for the pre-trained model.
4. **Response Formatting**: A module that maps the model predictions to meaningful labels (e.g., "setosa", "versicolor", "virginica") and formats the responses as JSON.

### Data Flow Diagram

```
+---------------+
|   API Client  |
+---------------+
        |
        | HTTP Request
        |
+---------------+
|   FastAPI     |
|  Application  |
+---------------+
        |
        | Input Data
        |
+---------------+
| Data          |
| Preprocessing |
+---------------+
        |
        | Preprocessed Data
        |
+---------------+
| Pre-trained   |
| SVC Model     |
+---------------+
        |
        | Predictions
        |
+---------------+
| Response      |
| Formatting    |
+---------------+
        |
        | JSON Response
        |
+---------------+
|   API Client  |
+---------------+
```

### API Specifications

The API provides the following endpoints:

1. **POST `/predict`**
   - Accepts sepal and petal measurements as input
   - Returns the predicted iris species

2. **POST `/predict_batch`**
   - Accepts a list of sepal and petal measurements
   - Returns a list of predicted iris species

3. **GET `/predict_random`**
   - Generates random sepal and petal measurements
   - Returns the predicted iris species for the random input

4. **GET `/health`**
   - Returns a "Healthy" status for health checking

5. **GET `/model_info`**
   - Returns information about the pre-trained model

6. **GET `/simulate_workload`**
   - Simulates a workload for a specified number of seconds (optional query parameter)

For detailed API documentation, including request/response formats and authentication methods, refer to the [API Documentation](#5-api-documentation) section.

### Database Schema

This application does not utilize a database. The pre-trained SVC model is loaded from a file during application startup.

## 3. Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/iris-classification-api.git
   ```

2. Navigate to the project directory:

   ```bash
   cd iris-classification-api
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Configuration Details

The application does not require any additional configuration. The pre-trained SVC model is loaded from the `svc_model.pkl` file in the project directory.

## 4. Code Structure

### Directory Organization

```
iris-classification-api/
├── iris_app.py
├── svc_model.pkl
├── README.md
├── requirements.txt
└── ...
```

- `iris_app.py`: The main application file containing the FastAPI application and API endpoints.
- `svc_model.pkl`: The pre-trained SVC model file.
- `README.md`: The project documentation file.
- `requirements.txt`: The file listing the project dependencies and their versions.

### Module Descriptions

- `iris_app.py`: This module contains the FastAPI application and defines the API endpoints for iris flower classification, health checking, model information retrieval, and workload simulation.

### Key Classes and Functions

- `app` (FastAPI instance): The main FastAPI application instance.
- `load_model()`: A function that loads the pre-trained SVC model from the `svc_model.pkl` file.
- `predict_iris_species()`: A function that takes sepal and petal measurements as input, preprocesses the data, and returns the predicted iris species using the pre-trained model.
- `predict_batch()`: A function that accepts a list of sepal and petal measurements, performs batch prediction using the pre-trained model, and returns a list of predicted iris species.
- `predict_random()`: A function that generates random sepal and petal measurements within typical ranges for iris flowers and returns the predicted iris species for the random input.
- `get_health()`: A function that returns a "Healthy" status for health checking.
- `get_model_info()`: A function that returns information about the pre-trained model, including its type, version, and a brief description.
- `simulate_workload()`: A function that simulates a workload for a specified number of seconds (optional query parameter), useful for testing latency and performance.

## 5. API Documentation

### Endpoint Descriptions

1. **POST `/predict`**
   - **Summary**: Predict Iris Species
   - **Description**: This endpoint accepts sepal and petal measurements as input and returns the predicted iris species.

2. **POST `/predict_batch`**
   - **Summary**: Batch Predict Iris Species
   - **Description**: This endpoint allows for batch prediction of multiple sets of sepal and petal measurements, returning a list of predicted iris species.

3. **GET `/predict_random`**
   - **Summary**: Get Random Iris Prediction
   - **Description**: This endpoint generates random sepal and petal measurements within typical ranges for iris flowers and returns the predicted iris species for the random input.

4. **GET `/health`**
   - **Summary**: Health Check
   - **Description**: This endpoint provides a simple health check for the application, returning a "Healthy" status.

5. **GET `/model_info`**
   - **Summary**: Get Model Information
   - **Description**: This endpoint returns information about the pre-trained model, including its type, version, and a brief description.

6. **GET `/simulate_workload`**
   - **Summary**: Simulate Workload
   - **Description**: This endpoint simulates a workload for a specified number of seconds (optional query parameter), useful for testing latency and performance.

### Request/Response Formats

1. **POST `/predict`**
   - **Request Body**:
     ```json
     {
       "sepal_length": float,
       "sepal_width": float,
       "petal_length": float,
       "petal_width": float
     }
     ```
   - **Response**:
     ```json
     {
       "prediction": "string (setosa, versicolor, virginica)",
       "code_signature": "string"
     }
     ```

2. **POST `/predict_batch`**
   - **Request Body**:
     ```json
     {
       "features": [
         {
           "sepal_length": float,
           "sepal_width": float,
           "petal_length": float,
           "petal_width": float
         },
         ...
       ]
     }
     ```
   - **Response**:
     ```json
     {
       "predictions": ["string (setosa, versicolor, virginica)", ...],
       "code_signature": "string"
     }
     ```

3. **GET `/predict_random`**
   - **Response**:
     ```json
     {
       "random_features": {
         "sepal_length": float,
         "sepal_width": float,
         "petal_length": float,
         "petal_width": float
       },
       "prediction": "string (setosa, versicolor, virginica)",
       "code_signature": "string"
     }
     ```

4. **GET `/health`**
   - **Response**:
     ```json
     {
       "status": "string",
       "code_signature": "string"
     }
     ```

5. **GET `/model_info`**
   - **Response**:
     ```json
     {
       "model": "string",
       "version": "string",
       "description": "string",
       "code_signature": "string"
     }
     ```

6. **GET `/simulate_workload`**
   - **Query Parameters**:
     - `seconds` (optional): An integer representing the number of seconds to simulate the workload.
   - **Response**:
     ```json
     {
       "message": "string",
       "code_signature": "string"
     }
     ```

### Authentication Methods

The Iris Flower Classification API does not require any authentication methods. All endpoints are publicly accessible.