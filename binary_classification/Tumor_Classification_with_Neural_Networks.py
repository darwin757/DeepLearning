import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 

# Load your dataset
file_path = 'tumor_classification_dataset.csv' 
data = pd.read_csv(file_path)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Visualization 1: Distribution of Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Target', data=data)
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Visualization 2: Correlation Matrix
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".1f")
plt.title('Correlation Matrix')
plt.show()

# Visualization 3: Feature Distributions - Example features
features_to_plot = ['Cell_Size', 'Cell_Shape', 'Texture']
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
for i, feature in enumerate(features_to_plot):
    sns.histplot(data[feature], ax=axs[i], kde=True)
    axs[i].set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# List of all features excluding the target
features = data.columns[:-1]  # Assuming the target is the last column

# Plot each feature in relation to the target variable
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Target', y=feature, data=data)
    plt.title(f'{feature} by Target')
    plt.xlabel('Target')
    plt.ylabel(feature)
    plt.show()

# Data Preparation with distinct training, validation, and testing sets
X = data.drop('Target', axis=1)  # Features
y = data['Target']  # Target variable

# First split: separate training set and temporary set (for further splitting)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: divide the temporary set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model Definition
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Model Compilation
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Training with validation set
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=10,
                    validation_data=(X_val_scaled, y_val), verbose=2)

# Final Model Evaluation using the test set
evaluation_results = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Final Test Loss: {evaluation_results[0]}, Final Test Accuracy: {evaluation_results[1]}")

# Predictions for Confusion Matrix on the test set
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.round(y_pred).astype(int)  # Convert probabilities to class labels

# Confusion Matrix Calculation
cm = confusion_matrix(y_test, y_pred_classes)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
