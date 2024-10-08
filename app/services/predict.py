import pickle
import pandas as pd
from app.schemas.customer import Customer

with open('resources/models/random_forest.pkl', 'rb') as f:
    modelo_rf = pickle.load(f)

with open('resources/models/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('resources/models/one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

def preprocess_data(df: pd.DataFrame):
    categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']
    df_encoded = ohe.transform(df[categorical_columns])
    df_encoded = pd.DataFrame(df_encoded, columns=ohe.get_feature_names_out(categorical_columns))

    df = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    numeric_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                       'Payment Delay', 'Total Spend', 'Last Interaction']
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df

def predict_churn(customer: Customer):
    data = {
        'Age': [customer.Age],
        'Gender': [customer.Gender],
        'Tenure': [customer.Tenure],
        'Usage Frequency': [customer.UsageFrequency],
        'Support Calls': [customer.SupportCalls],
        'Payment Delay': [customer.PaymentDelay],
        'Subscription Type': [customer.SubscriptionType],
        'Contract Length': [customer.ContractLength],
        'Total Spend': [customer.TotalSpend],
        'Last Interaction': [customer.LastInteraction]
    }
    df_input = pd.DataFrame(data)
    df_processed = preprocess_data(df_input)
    prediction = modelo_rf.predict(df_processed)
    return int(prediction[0])
