import joblib
import os
from django.conf import settings

# Function to load models from a specific folder
def load_models(folder_name):
    # Path to the folder containing the .pkl files
    MODELS_DIR = os.path.join(settings.BASE_DIR, folder_name)

    # Load label encoders (if applicable)
    le_operator = joblib.load(os.path.join(MODELS_DIR, 'le_operator.pkl')) if os.path.exists(os.path.join(MODELS_DIR, 'le_operator.pkl')) else None
    le_network = joblib.load(os.path.join(MODELS_DIR, 'le_network.pkl')) if os.path.exists(os.path.join(MODELS_DIR, 'le_network.pkl')) else None
    le_state = joblib.load(os.path.join(MODELS_DIR, 'le_state.pkl')) if os.path.exists(os.path.join(MODELS_DIR, 'le_state.pkl')) else None

    # Load scaler
    minmax_scaler = joblib.load(os.path.join(MODELS_DIR, 'minmax_scaler1.pkl'))

    # Load models
    xgboost_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_modelv1.pkl'))
    lstm_model = joblib.load(os.path.join(MODELS_DIR, 'lstmv1.pkl'))
    mlp_model = joblib.load(os.path.join(MODELS_DIR, 'MLPv1.pkl'))

    return {
        'le_operator': le_operator,
        'le_network': le_network,
        'le_state': le_state,
        'minmax_scaler': minmax_scaler,
        'xgboost_model': xgboost_model,
        'lstm_model': lstm_model,
        'mlp_model': mlp_model,
    }
    
def load_predictive_maintenance_models():
    MODELS_DIR = os.path.join(settings.BASE_DIR, 'predictive_maintenance')
    failure_model = joblib.load(os.path.join(MODELS_DIR, 'predictive_maintenance_model.pkl'))
    return {'failure_model': failure_model}

def load_cooling_system_models():
    MODELS_DIR = os.path.join(settings.BASE_DIR, 'cooling_system')
    tmax_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_modelMaxTemp.pkl'))
    tmin_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model_MinTemp.pkl'))
    return {'tmax_model': tmax_model, 'tmin_model': tmin_model}