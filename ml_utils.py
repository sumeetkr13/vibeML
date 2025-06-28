import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

def detect_problem_type(target_series):
    """
    Automatically detect if the problem is classification or regression
    based on the target variable.
    """
    # Remove missing values for analysis
    target_clean = target_series.dropna()
    
    # If target is object/string type, it's classification
    if target_clean.dtype == 'object':
        return "classification"
    
    # If target is numeric, check unique values
    unique_values = target_clean.nunique()
    total_values = len(target_clean)
    
    # If unique values are less than 10% of total values and less than 20, classify as classification
    if unique_values < 20 and unique_values / total_values < 0.1:
        return "classification"
    
    # Check if all values are integers (could be classification)
    if target_clean.dtype in ['int64', 'int32'] and unique_values <= 10:
        return "classification"
    
    return "regression"

def handle_missing_values(df, strategies):
    """
    Handle missing values based on user-defined strategies for each column.
    
    Parameters:
    df: pandas DataFrame
    strategies: dict mapping column names to strategies
    """
    df_processed = df.copy()
    
    for column, strategy in strategies.items():
        if column not in df_processed.columns:
            continue
            
        if strategy == "Drop rows":
            df_processed = df_processed.dropna(subset=[column])
        elif strategy == "Fill with mean":
            if pd.api.types.is_numeric_dtype(df_processed[column]):
                df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
        elif strategy == "Fill with median":
            if pd.api.types.is_numeric_dtype(df_processed[column]):
                df_processed[column] = df_processed[column].fillna(df_processed[column].median())
        elif strategy == "Fill with mode":
            mode_value = df_processed[column].mode()
            if len(mode_value) > 0:
                df_processed[column] = df_processed[column].fillna(mode_value[0])
        elif strategy == "Forward fill":
            df_processed[column] = df_processed[column].fillna(method='ffill')
    
    return df_processed

def prepare_features(df, feature_columns, target_column):
    """
    Prepare features for machine learning by handling categorical variables.
    """
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return X, y, preprocessor, categorical_features, numerical_features

def get_model(model_code, problem_type):
    """
    Get the appropriate model based on model code and problem type.
    """
    if problem_type == "classification":
        models = {
            "rf": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2),
            "lr": LogisticRegression(random_state=42, max_iter=1000),
            "svm": SVC(random_state=42, probability=True),
            "gb": GradientBoostingClassifier(random_state=42, n_estimators=100),
            "knn": KNeighborsClassifier(n_neighbors=5)
        }
    else:  # regression
        models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2),
            "lr": LinearRegression(),
            "svr": SVR(kernel='rbf'),
            "gb": GradientBoostingRegressor(random_state=42, n_estimators=100),
            "knn": KNeighborsRegressor(n_neighbors=5)
        }
    
    return models.get(model_code, models["rf"])  # Default to Random Forest

def train_model(df, target_column, feature_columns, problem_type, model_code="rf", test_size=0.2):
    """
    Train a machine learning model based on the problem type and selected model.
    """
    # Prepare features
    X, y, preprocessor, categorical_features, numerical_features = prepare_features(
        df, feature_columns, target_column
    )
    
    # Handle target variable for classification
    label_encoder = None
    if problem_type == "classification" and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "classification" else None
    )
    
    # Get the selected model
    model = get_model(model_code, problem_type)
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Transform data for evaluation
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    return pipeline, X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, label_encoder

def evaluate_model(y_true, y_pred, problem_type):
    """
    Evaluate model performance based on problem type.
    """
    metrics = {}
    
    if problem_type == "classification":
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        try:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except TypeError:
            # Fallback for older sklearn versions
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    else:  # regression
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    return metrics

def get_feature_importance(model, feature_names, preprocessor):
    """
    Extract feature importance from the trained model.
    """
    # Get the actual model from pipeline
    rf_model = model.named_steps['model']
    
    # Get feature importance
    importance = rf_model.feature_importances_
    
    # Get feature names after preprocessing
    try:
        # Try to get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names_transformed = preprocessor.get_feature_names_out()
        else:
            # Fallback: use original feature names
            feature_names_transformed = feature_names
    except:
        # If preprocessing fails, use original feature names
        feature_names_transformed = feature_names[:len(importance)]
    
    # Create feature importance dictionary
    if len(feature_names_transformed) != len(importance):
        # Handle mismatch in feature names
        feature_names_transformed = [f"Feature_{i}" for i in range(len(importance))]
    
    importance_dict = dict(zip(feature_names_transformed, importance))
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict
