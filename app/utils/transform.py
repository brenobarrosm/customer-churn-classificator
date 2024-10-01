import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Inicializar OneHotEncoder e StandardScaler globalmente para reutilização
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
scaler = StandardScaler()


def preprocess_data(df: pd.DataFrame):
    # One-Hot Encoding
    categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']
    df_encoded = ohe.fit_transform(df[categorical_columns]).toarray()
    df_encoded = pd.DataFrame(df_encoded, columns=ohe.get_feature_names_out())

    # Combinar dados codificados com o DataFrame original, removendo as colunas categóricas
    df = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    # Normalização
    numeric_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend',
                       'Last Interaction']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df
