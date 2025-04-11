from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # For loading the LSTM model
import joblib

# Global dictionary to store loaded models and encoders
reduce_models = {}

def load_models():
    """
    Load all models, scaler, and encoders from the reducing_subcarriers folder.
    """
    try:
        # Load MinMaxScaler
        reduce_models['minmax_scaler'] = joblib.load('reducing_subcarriers/minmax_scaler1.pkl')

        # Load Label Encoders
        reduce_models['le_operator'] = joblib.load('reducing_subcarriers/le_operator.pkl')
        reduce_models['le_network'] = joblib.load('reducing_subcarriers/le_network.pkl')
        reduce_models['le_state'] = joblib.load('reducing_subcarriers/le_state.pkl')

        # Load Models
        reduce_models['xgboost_model'] = joblib.load('reducing_subcarriers/xgboost_modelv1.pkl')

        # Debugging: Check if the LSTM model file exists
        # lstm_model_path = 'reducing_subcarriers/lstmv1.pkl'
        # if not os.path.exists(lstm_model_path):
        #     print(f"Error: LSTM model file not found at {lstm_model_path}")
        # else:
        #     reduce_models['lstm_model'] = load_model(lstm_model_path)  # TensorFlow/Keras model
        #     print("LSTM model loaded successfully.")

        reduce_models['lstm_model'] = joblib.load('reducing_subcarriers/lstmv1.pkl')
        reduce_models['mlp_model'] = joblib.load('reducing_subcarriers/MLPv1.pkl')

        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models at startup
load_models()
# Predictive maintenance models
maintenance_models = {}

def load_maintenance_models():
    try:
        maintenance_models['model'] = joblib.load('predictive_maintenance/predictive_maintenance/model.pkl')
        maintenance_models['scaler'] = joblib.load('predictive_maintenance/predictive_maintenance/scaler.pkl')
        maintenance_models['type_encoder'] = joblib.load('predictive_maintenance/predictive_maintenance/type_encoder.pkl')
        maintenance_models['product_id_encoder'] = joblib.load('predictive_maintenance/predictive_maintenance/product_id_encoder.pkl')
        maintenance_models['failure_type_encoder'] = joblib.load('predictive_maintenance/predictive_maintenance/failure_type_encoder.pkl')
        
        
        # Load features list
        with open('predictive_maintenance/predictive_maintenance/features.txt') as f:
            maintenance_models['features'] = f.read().splitlines()

        print("Predictive maintenance model loaded!")
    except Exception as e:
        print(f"Error loading maintenance models: {e}")

# Load them at startup
load_maintenance_models()

@csrf_exempt
def reduce(request):
    if request.method == 'POST':
        try:
            # Parse input JSON
            data = json.loads(request.body)

            # Extract input data (all fields except 'label')
            input_data = {
                "Longitude": float(data.get("Longitude")),
                "Latitude": float(data.get("Latitude")),
                "Speed": float(data.get("Speed")),
                "Operatorname": data.get("Operatorname"),
                "CellID": int(data.get("CellID")),
                "NetworkMode": data.get("NetworkMode"),
                "RSRP": float(data.get("RSRP")),
                "RSRQ": float(data.get("RSRQ")),
                "SNR": float(data.get("SNR")),
                "CQI": int(data.get("CQI")),
                "RSSI": float(data.get("RSSI")),
                "DL_bitrate": float(data.get("DL_bitrate")),
                "UL_bitrate": float(data.get("UL_bitrate")),
                "State": data.get("State"),
                "NRxRSRP": float(data.get("NRxRSRP")),
                "NRxRSRQ": float(data.get("NRxRSRQ")),
                "ServingCell_Lon": float(data.get("ServingCell_Lon")),
                "ServingCell_Lat": float(data.get("ServingCell_Lat")),
                "ServingCell_Distance": float(data.get("ServingCell_Distance")),
                "year": int(data.get("year")),
                "month": int(data.get("month")),
                "day": int(data.get("day")),
                "hour": int(data.get("hour")),
                "minute": int(data.get("minute")),
                "second": int(data.get("second")),
            }

            # Convert input data to DataFrame
            df = pd.DataFrame([input_data])

            # Apply label encoding (if applicable)
            if reduce_models['le_operator']:
                df['Operatorname'] = reduce_models['le_operator'].transform(df['Operatorname'])
            if reduce_models['le_network']:
                df['NetworkMode'] = reduce_models['le_network'].transform(df['NetworkMode'])
            if reduce_models['le_state']:
                df['State'] = reduce_models['le_state'].transform(df['State'])

            # Scale features
            scaled_features = reduce_models['minmax_scaler'].transform(df)

            # Reshape for LSTM (samples, timesteps, features)
            lstm_input = np.expand_dims(scaled_features, axis=1)

            # Predict using each model
            xgboost_prediction = reduce_models['xgboost_model'].predict(scaled_features)[0]
            lstm_prediction = reduce_models['lstm_model'].predict(lstm_input).flatten()[0] >= 0.5
            mlp_prediction = reduce_models['mlp_model'].predict(scaled_features)[0]

            # Convert predictions to integers
            xgboost_prediction = int(xgboost_prediction)
            lstm_prediction = int(lstm_prediction)
            mlp_prediction = int(mlp_prediction)

            # Implement voting logic
            predictions = [xgboost_prediction, lstm_prediction, mlp_prediction]
            unique_predictions = set(predictions)

            if len(unique_predictions) == 1:
                # All models agree
                final_prediction = predictions[0]
            else:
                # Disagreement: prioritize XGBoost
                final_prediction = xgboost_prediction

            # Return response
            return JsonResponse({
                "status": "success",
                "prediction": final_prediction,
                "details": {
                    "xgboost_prediction": xgboost_prediction,
                    "lstm_prediction": lstm_prediction,
                    "mlp_prediction": mlp_prediction,
                    "final_prediction": final_prediction
                }
            })

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)


# Cooling System Prediction
cooling_system_models = {}

def load_cooling_system_models():
    """
    Load cooling system models.
    """
    try:
        cooling_system_models['tmax_model'] = joblib.load('cooling_system/xgboost_modelMaxTemp.pkl')
        cooling_system_models['tmin_model'] = joblib.load('cooling_system/xgboost_model_MinTemp.pkl')
        print("Cooling system models loaded successfully!")
    except Exception as e:
        print(f"Error loading cooling system models: {e}")

# Load cooling system models at startup
load_cooling_system_models()

@csrf_exempt
def cool(request):
    if request.method == 'POST':
        try:
            # Parse input JSON
            data = json.loads(request.body)

            # Example input structure
            input_data = {
                "station": data.get("station"),
                "Present_Tmax": float(data.get("Present_Tmax")),
                "Present_Tmin": float(data.get("Present_Tmin")),
                "LDAPS_RHmin": float(data.get("LDAPS_RHmin")),
                "LDAPS_RHmax": float(data.get("LDAPS_RHmax")),
                "LDAPS_Tmax_lapse": float(data.get("LDAPS_Tmax_lapse")),
                "LDAPS_Tmin_lapse": float(data.get("LDAPS_Tmin_lapse")),
                "LDAPS_WS": float(data.get("LDAPS_WS")),
                "LDAPS_LH": float(data.get("LDAPS_LH")),
                "LDAPS_CC1": float(data.get("LDAPS_CC1")),
                "LDAPS_CC2": float(data.get("LDAPS_CC2")),
                "LDAPS_CC3": float(data.get("LDAPS_CC3")),
                "LDAPS_CC4": float(data.get("LDAPS_CC4")),
                "LDAPS_PPT1": float(data.get("LDAPS_PPT1")),
                "LDAPS_PPT2": float(data.get("LDAPS_PPT2")),
                "LDAPS_PPT3": float(data.get("LDAPS_PPT3")),
                "LDAPS_PPT4": float(data.get("LDAPS_PPT4")),
                "lat": float(data.get("lat")),
                "lon": float(data.get("lon")),
                "DEM": float(data.get("DEM")),
                "Slope": float(data.get("Slope")),
                "Solar radiation": float(data.get("Solar radiation")),
                "year": int(data.get("year")),
                "month": int(data.get("month")),
                "day": int(data.get("day")),
            }

            # Convert input data to DataFrame
            df = pd.DataFrame([input_data])

            # Predict Next_Tmax and Next_Tmin
            tmax_model = cooling_system_models['tmax_model']
            tmin_model = cooling_system_models['tmin_model']
            next_tmax = tmax_model.predict(df)[0]  # Predicted value
            next_tmin = tmin_model.predict(df)[0]  # Predicted value

            # Ensure predictions are JSON-serializable
            next_tmax = np.float32(next_tmax).item()  # Convert to Python float
            next_tmin = np.float32(next_tmin).item()  # Convert to Python float

            # Return response
            return JsonResponse({
                "status": "success",
                "next_tmax": next_tmax,
                "next_tmin": next_tmin,
            })

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)


    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_df = pd.DataFrame([data])
            
            # Encode categorical variables safely
            def safe_transform(encoder, value):
                if value in encoder.classes_:
                    return encoder.transform([value])[0]
                else:
                    return -1  # Default value for unseen labels
            
            input_df['Type'] = safe_transform(maintenance_models['type_encoder'], input_df['Type'].iloc[0])
            input_df['Product_ID'] = safe_transform(maintenance_models['product_id_encoder'], input_df['Product_ID'].iloc[0])
            
            # Scale numerical features
            numericals = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']
            input_df[numericals] = maintenance_models['scaler'].transform(input_df[numericals])
            
            # Align columns with training data
            input_df = input_df[maintenance_models['features']]
            
            # Predict probabilities
            probs = maintenance_models['model'].predict_proba(input_df)[0]
            
            # Map to original class labels using inverse_transform
            failure_labels = maintenance_models['failure_type_encoder'].classes_
            failure_mapping = {
                int(i): str(label) for i, label in enumerate(failure_labels)  # Ensure keys are int
            }
            
            # Clip and renormalize probabilities
            smoothing = 0.01
            probs = np.clip(probs, smoothing, 1 - smoothing)
            probs /= probs.sum()
            
            # Prepare result
            result = {
                "status": "success",
                "failure_probabilities": {
                    str(failure_mapping[int(i)]): round(float(prob) * 100, 2)  # Ensure keys are str
                    for i, prob in enumerate(probs)
                },
                "most_likely_failure": str(failure_mapping[int(np.argmax(probs))]),  # Ensure key is int
                "max_probability": round(float(np.max(probs)) * 100, 2)
            }
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)