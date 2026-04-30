# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import kagglehub

# ================================
# 2. Load Dataset
# ================================
path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")

# Find CSV file
for file in os.listdir(path):
    if file.endswith(".csv"):
        file_path = os.path.join(path, file)

print("Dataset path:", file_path)

df = pd.read_csv(file_path)

print("\nDataset Shape:", df.shape)
print(df.head())


# ================================
# 3. Data Cleaning
# ================================

# Drop columns with too many missing values or not useful
df.drop(['company', 'agent'], axis=1, inplace=True)

# Fill missing values
df['country'].fillna(df['country'].mode()[0], inplace=True)
df['children'].fillna(0, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nAfter cleaning:", df.shape)


# ================================
# 4. Feature Engineering
# ================================

# Convert categorical variables using Label Encoding
le = LabelEncoder()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Target variable
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']


# ================================
# 5. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# 6. Model Training
# ================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ================================
# 7. Predictions
# ================================
y_pred = model.predict(X_test)


# ================================
# 8. Evaluation
# ================================
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ================================
# 9. Feature Importance (Insights)
# ================================
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:\n")
print(feature_importance_df.head(10))


# ================================
# 10. Example Prediction
# ================================
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("\nSample Prediction (0 = Not Canceled, 1 = Canceled):", prediction[0])
print("Cancellation Probability:", probability[0][1])