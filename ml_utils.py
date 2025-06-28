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
    print(f"Debug: train_model called with df type: {type(df)}")
    print(f"Debug: feature_columns type: {type(feature_columns)}, value: {feature_columns}")
    print(f"Debug: target_column: {target_column}")
    
    # Ensure df is a DataFrame and feature_columns is a list
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    
    print(f"Debug: About to select features from DataFrame")
    print(f"Debug: DataFrame columns: {list(df.columns)}")
    print(f"Debug: Selected features: {feature_columns}")
    
    # Prepare features and target - keep as DataFrame/Series
    try:
        X = df[feature_columns].copy()
        print(f"Debug: X created successfully, type: {type(X)}, shape: {X.shape}")
    except Exception as e:
        print(f"Debug: Error creating X: {e}")
        raise
    
    try:
        y = df[target_column].copy()
        print(f"Debug: y created successfully, type: {type(y)}, shape: {y.shape}")
    except Exception as e:
        print(f"Debug: Error creating y: {e}")
        raise
    
    # Handle target variable for classification
    label_encoder = None
    if problem_type == "classification" and y.dtype == 'object':
        print(f"Debug: Encoding target variable")
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)
        print(f"Debug: Target encoded, new type: {type(y)}")
    
    print(f"Debug: About to split data")
    # Generate train/test indices to maintain DataFrame structure
    try:
        train_idx, test_idx = train_test_split(
            df.index, test_size=test_size, random_state=42, 
            stratify=y if problem_type == "classification" else None
        )
        print(f"Debug: Split successful, train indices: {len(train_idx)}, test indices: {len(test_idx)}")
    except Exception as e:
        print(f"Debug: Error in train_test_split: {e}")
        raise
    
    # Use index slicing to keep DataFrames
    print(f"Debug: Creating train/test sets")
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()
    print(f"Debug: Train/test sets created. X_train type: {type(X_train)}, shape: {X_train.shape}")
    
    # Identify categorical and numerical columns from the original DataFrame
    print(f"Debug: Identifying feature types")
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Debug: Categorical features: {categorical_features}")
    print(f"Debug: Numerical features: {numerical_features}")
    
    # Create preprocessing pipeline
    print(f"Debug: Creating preprocessor")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    print(f"Debug: Preprocessor created")
    
    # Get the selected model
    print(f"Debug: Getting model: {model_code}")
    model = get_model(model_code, problem_type)
    print(f"Debug: Model created: {type(model)}")
    
    # Create pipeline with preprocessing
    print(f"Debug: Creating pipeline")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    print(f"Debug: Pipeline created")
    
    # Fit the pipeline - X_train is still a DataFrame here
    print(f"Debug: About to fit pipeline. X_train type: {type(X_train)}")
    print(f"Debug: X_train columns: {list(X_train.columns)}")
    try:
        pipeline.fit(X_train, y_train)
        print(f"Debug: Pipeline fitted successfully")
    except Exception as e:
        print(f"Debug: Error fitting pipeline: {e}")
        raise
    
    # Transform data for evaluation (using the fitted pipeline's preprocessor)
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Return original DataFrames for X_train and X_test, not the transformed arrays
    # The pipeline will handle the transformation internally when needed
    return pipeline, X_train, X_test, y_train, y_test, preprocessor, label_encoder

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
    print(f"Debug: get_feature_importance called")
    print(f"Debug: feature_names type: {type(feature_names)}, value: {feature_names}")
    
    # Get the actual model from pipeline
    ml_model = model.named_steps['model']
    print(f"Debug: Model extracted: {type(ml_model)}")
    
    # Check if model has feature importance
    if not hasattr(ml_model, 'feature_importances_'):
        print(f"Debug: Model does not have feature_importances_")
        return {}
    
    # Get feature importance
    importance = ml_model.feature_importances_
    print(f"Debug: Feature importance shape: {importance.shape}")
    
    # Get feature names after preprocessing
    try:
        print(f"Debug: Trying to get feature names from preprocessor")
        # Try to get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            print(f"Debug: Preprocessor has get_feature_names_out method")
            feature_names_transformed = preprocessor.get_feature_names_out()
            print(f"Debug: Got transformed feature names: {feature_names_transformed}")
        else:
            print(f"Debug: Preprocessor does not have get_feature_names_out, using original")
            # Fallback: use original feature names
            feature_names_transformed = feature_names
    except Exception as e:
        print(f"Debug: Error getting feature names from preprocessor: {e}")
        # If preprocessing fails, use original feature names
        feature_names_transformed = feature_names[:len(importance)]
    
    print(f"Debug: Final feature names length: {len(feature_names_transformed)}, importance length: {len(importance)}")
    
    # Create feature importance dictionary
    if len(feature_names_transformed) != len(importance):
        print(f"Debug: Mismatch in lengths, creating generic names")
        # Handle mismatch in feature names
        feature_names_transformed = [f"Feature_{i}" for i in range(len(importance))]
    
    importance_dict = dict(zip(feature_names_transformed, importance))
    print(f"Debug: Created importance dict with {len(importance_dict)} items")
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    return importance_dict
