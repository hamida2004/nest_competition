import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
import collections

# Create directory for saving models if it doesn't exist
os.makedirs('predictive_maintenance', exist_ok=True)

# Step 1: Load the dataset
data = pd.read_csv("dataset_with_failure_type.csv")

# Step 2: Preprocess the data
data = data.drop(columns=["UDI"], errors="ignore")

# Encode categorical variables
label_encoders = {}
for column in ["Type", "Product_ID"]:
    if column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
        if column == "Type":
            joblib.dump(le, 'predictive_maintenance/type_encoder.pkl')
        elif column == "Product_ID":
            joblib.dump(le, 'predictive_maintenance/product_id_encoder.pkl')

# Encode target variable: Failure_Type
failure_type_encoder = LabelEncoder()
data['Failure_Type'] = failure_type_encoder.fit_transform(data['Failure_Type'])
joblib.dump(failure_type_encoder, 'predictive_maintenance/failure_type_encoder.pkl')

# Print original class distribution
print("\nOriginal Failure_Type distribution:")
print(data['Failure_Type'].value_counts())

# Normalize numerical features
numerical_features = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
joblib.dump(scaler, 'predictive_maintenance/scaler.pkl')

# Step 3: Feature selection
X = data.drop(columns=['Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Failure_Type'], errors="ignore")
y = data['Failure_Type']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print new class distribution after SMOTE
print("\nFailure_Type distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Step 6: Train the model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(X_train_resampled, y_train_resampled)
joblib.dump(model, 'predictive_maintenance/model.pkl')

# Save feature names
with open('predictive_maintenance/features.txt', 'w') as f:
    f.write('\n'.join(X.columns.tolist()))

# Map integer labels back to names for display
failure_mapping = {
    int(i): str(label) for i, label in enumerate(failure_type_encoder.classes_)
}

# Single prediction function
def predict_single(input_data):
    try:
        input_df = pd.DataFrame([input_data])

        # Safely transform categorical variables
        def safe_transform(encoder, value):
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                return -1  # Default value for unseen labels

        input_df['Type'] = safe_transform(label_encoders['Type'], input_data['Type'])
        input_df['Product_ID'] = safe_transform(label_encoders['Product_ID'], input_data['Product_ID'])

        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Align columns with training data
        input_df = input_df[X.columns]

        # Predict probabilities
        probs = model.predict_proba(input_df)[0]

        # Clip and renormalize probabilities
        smoothing = 0.01
        probs = np.clip(probs, smoothing, 1 - smoothing)
        probs /= probs.sum()

        # Prepare result
        result = {
            "status": "success",
            "failure_probabilities": {
                str(failure_mapping[int(i)]): round(float(prob) * 100, 2)
                for i, prob in enumerate(probs)
            },
            "most_likely_failure": str(failure_mapping[int(np.argmax(probs))]),
            "max_probability": round(float(np.max(probs)) * 100, 2)
        }
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Step 7: Evaluate model on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on original test set: {accuracy * 100:.2f}%")


# عرض أهمية كل ميزة
importance = model.feature_importances_
features = X.columns

for f, imp in zip(features, importance):
    print(f"{f}: {imp:.4f}")

# رسم بياني
plt.figure(figsize=(10, 5))
plt.barh(features, importance)
plt.title("Feature Importances")
plt.show()