import pandas as pd

# Load the dataset
data = pd.read_csv('C:\\Users\\Asus\\Documents\\Downloads\\Dataset.csv')

# Display the first few rows
print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into features and target
X = data.drop('Type', axis=1)  # Assuming 'target' is the name of the target column
y = data['Type']

# Handle missing values if any
X = X.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Initialize a more complex model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

import joblib

# Save the logistic regression model
joblib.dump(model, 'logistic_model.pkl')

# Save the random forest model
joblib.dump(rf_model, 'rf_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
