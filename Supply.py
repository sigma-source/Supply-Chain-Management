import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('supply_chain_data.csv')

# Display first few rows to understand the structure
data.head()

# Data Cleaning
# Handle missing values
data = data.dropna()

# Feature Engineering
# Extract numerical features for analysis
data['Profit Margin'] = data['Revenue generated'] - data['Manufacturing costs'] - data['Shipping costs']

# Normalize numerical columns
num_cols = ['Price', 'Availability', 'Number of products sold', 'Revenue generated', 'Stock levels',
            'Lead times', 'Order quantities', 'Shipping times', 'Shipping costs', 'Manufacturing costs',
            'Production volumes']
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# One-hot encode categorical features
cat_cols = ['Product type', 'Customer demographics', 'Supplier name', 'Location', 'Transportation modes', 'Shipping carriers', 'Routes']
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Define features and target
X = data.drop(columns=['SKU', 'Profit Margin', 'Defect rates', 'Inspection results'])
y = data['Profit Margin']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=1, callbacks=[early_stopping])

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Visualizations
# Loss Curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Predictions vs Actual Values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Profit Margin')
plt.show()

# Feature Importance Analysis (Correlation Matrix)
plt.figure(figsize=(12, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Save the model
model.save('supply_chain_profit_model.h5')

print("Model training and evaluation complete. The trained model has been saved for future use.")
