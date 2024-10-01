import pickle
import pandas as pd
from app.schemas.customer import Customer
from sklearn.preprocessing import OneHotEncoder

# Carregar o modelo salvo com pickle
with open('resources/models/random_forest.pkl', 'rb') as f:
    modelo_rf = pickle.load(f)

# Carregar o StandardScaler salvo
with open('resources/models/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Inicializar OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)


def preprocess_data(df: pd.DataFrame):
    # Aplicar One-Hot Encoding nas colunas categóricas
    categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']
    df_encoded = ohe.fit_transform(df[categorical_columns])
    df_encoded = pd.DataFrame(df_encoded.toarray(), columns=ohe.get_feature_names_out())

    # Concatenar com o DataFrame original e remover as colunas originais
    df = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    # Normalizar as colunas numéricas usando o StandardScaler carregado
    numeric_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                       'Payment Delay', 'Total Spend', 'Last Interaction']
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df


def predict_churn(customer: Customer):
    # Converter o Customer (Pydantic Model) para um DataFrame
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

    # Pré-processar os dados
    df_processed = preprocess_data(df_input)

    # Fazer a predição
    prediction = modelo_rf.predict(df_processed)

    # Retornar o resultado da predição
    return int(prediction[0])
