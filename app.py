import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = '/home/breno/PycharmProjects/customer-churn-classificator/resources'

df = pd.read_csv(f'{BASE_DIR}/output/prepared_data.csv')
df.dropna(inplace=True)
X = df.drop(columns='Churn')
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

with open(f'{BASE_DIR}/models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
