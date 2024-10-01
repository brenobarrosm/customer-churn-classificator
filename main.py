from fastapi import FastAPI
from app.services.predict import predict_churn
from app.schemas.customer import Customer

app = FastAPI()

@app.post("/predict/")
def predict_churn_api(customer: Customer):
    prediction = predict_churn(customer)
    return {"CustomerID": customer.CustomerID, "Churn_Prediction": prediction}
