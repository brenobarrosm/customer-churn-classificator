import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Carregar o modelo salvo com pickle
with open('resources/models/random_forest.pkl', 'rb') as f:
    modelo_rf = pickle.load(f)

# Carregar o OneHotEncoder salvo
with open('resources/models/one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

# Dados de entrada em formato JSON
input_data = {
    "Age": 30,
    "Gender": "Female",
    "Tenure": 24,
    "Usage Frequency": 5,
    "Support Calls": 1,
    "Payment Delay": 0,
    "Subscription Type": "Standard",
    "Contract Length": "Monthly",
    "Total Spend": 1200.50,
    "Last Interaction": 7
}

# Converter o JSON para DataFrame
df_input = pd.DataFrame([input_data])

# Pré-processar os dados
# Aplicar One Hot Encoding
df_encoded = ohe.transform(df_input[['Gender', 'SubscriptionType', 'ContractLength']]).toarray()
df_encoded = pd.DataFrame(df_encoded, columns=ohe.get_feature_names_out())

df_processed = pd.concat([df_input.drop(['Gender', 'SubscriptionType', 'ContractLength'], axis=1), df_encoded], axis=1)

numeric_columns = ['Age', 'Tenure', 'UsageFrequency', 'SupportCalls', 'PaymentDelay', 'TotalSpend', 'LastInteraction']
scaler = StandardScaler()

df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

prediction = modelo_rf.predict(df_processed)

print("Predição de Churn:", int(prediction[0]))
