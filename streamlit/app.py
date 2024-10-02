import streamlit as st
import requests

# URL da API (modifique para o endereço correto)
API_URL = "http://localhost:8000/predict/"  # Alterar para o URL real da sua API


# Função para enviar os dados para a API e obter a predição
def get_prediction(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erro: {response.status_code}")
        return None


# Interface do Streamlit
st.title("Previsão de Desistência de Clientes")

# Coletar as entradas do usuário
customer_id = st.number_input("Customer ID", min_value=1, step=1)
age = st.number_input("Age", min_value=0, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
tenure = st.number_input("Tenure", min_value=0, step=1)
usage_frequency = st.number_input("Usage Frequency", min_value=0, step=1)
support_calls = st.number_input("Support Calls", min_value=0, step=1)
payment_delay = st.number_input("Payment Delay", min_value=0, step=1)
subscription_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend ($)", min_value=0.0, step=0.01, format="%.2f")
last_interaction = st.number_input("Last Interaction (Days)", min_value=0, step=1)

# Botão para submeter os dados
if st.button("Prever Desistência"):
    # Construir o payload JSON
    data = {
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "UsageFrequency": usage_frequency,
        "SupportCalls": support_calls,
        "PaymentDelay": payment_delay,
        "SubscriptionType": subscription_type,
        "ContractLength": contract_length,
        "TotalSpend": total_spend,
        "LastInteraction": last_interaction
    }

    # Fazer a requisição para a API
    result = get_prediction(data)

    # Mostrar o resultado
    if result:
        if result["Churn_Prediction"] == 1:
            st.success(f"O cliente ID {customer_id} provavelmente irá desistir.")
        else:
            st.success(f"O cliente ID {customer_id} provavelmente **não** irá desistir.")

