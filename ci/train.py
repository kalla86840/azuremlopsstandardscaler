import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, '..', 'data', 'cars_dataset_no_brand 300 rows.csv'))

X = df.drop('SalePrice_USD', axis=1).select_dtypes(include=['number'])
y = df['SalePrice_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved.")
