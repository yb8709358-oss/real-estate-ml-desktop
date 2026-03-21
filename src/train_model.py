import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
if not os.path.exists('models'):
    os.makedirs('models')

# 1. data loading
df = pd.read_csv('data/Housing.csv') 

# 2. data preparation (must match the format expected by the model)
# Ensure all categorical columns are converted to 0 and 1 as done previously
# ... (add conversion code here) ...

X = df.drop('price', axis=1)
y = df['price']

# 3. model training
model = LinearRegression() # create the model
# 1. data loading
df = pd.read_csv('data/Housing.csv')

# 2. data preparation (must match the format expected by the model)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# to convert furnishingstatus into binary features, we can use get_dummies
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Now we have the data ready for training
X = df.drop('price', axis=1)
y = df['price']

# 3. model training
model = LinearRegression()
model.fit(X, y)
model.fit(X, y)            # train the model 

# 4. Save the model (make sure the 'models' directory exists in your project)
joblib.dump(model, 'models/house_model.pkl')
print("Model trained and saved successfully!")