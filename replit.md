# ML CSV Analyzer

## Overview

This is a Streamlit-based machine learning application that provides automated CSV data analysis and model training capabilities. The application automatically detects whether a dataset requires classification or regression modeling, handles data preprocessing, trains models, and provides comprehensive visualizations and performance metrics.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for user interaction
- **ML Engine**: Scikit-learn based machine learning pipeline with automated model selection
- **Visualization Engine**: Matplotlib/Seaborn based plotting system
- **Data Processing**: Pandas-based data manipulation and preprocessing

The architecture is designed as a single-page application with session state management for maintaining user progress through the ML workflow.

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Primary Streamlit interface and workflow orchestration
- **Key Features**:
  - Session state management for multi-step ML workflow
  - File upload and data preview functionality
  - Step-by-step guided ML pipeline execution
  - Download functionality for results

### 2. ML Utilities (`ml_utils.py`)
- **Purpose**: Core machine learning functionality
- **Key Features**:
  - Automatic problem type detection (classification vs regression)
  - Missing value handling with multiple strategies
  - Model training using Random Forest algorithms
  - Model evaluation with comprehensive metrics
  - Feature importance extraction

### 3. Visualization Utilities (`visualization_utils.py`)
- **Purpose**: Data visualization and model performance plotting
- **Key Features**:
  - Confusion matrix visualization for classification
  - Feature importance plots
  - Performance metric visualizations
  - Data overview and exploratory plots

## Data Flow

1. **Data Upload**: User uploads CSV file through Streamlit interface
2. **Data Exploration**: Application displays data overview and basic statistics
3. **Problem Detection**: Automatic detection of classification vs regression based on target variable
4. **Data Preprocessing**: Handling of missing values and feature encoding
5. **Model Training**: Random Forest model training with automated hyperparameter selection
6. **Model Evaluation**: Comprehensive performance metrics and visualizations
7. **Results Export**: Download capabilities for processed data and model results

## External Dependencies

### Core ML Stack
- **scikit-learn**: Primary machine learning library for model training and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing foundation

### Visualization Stack
- **matplotlib**: Core plotting library
- **seaborn**: Statistical data visualization

### Web Framework
- **streamlit**: Web application framework for the user interface

### Data Processing
- **StandardScaler**: Feature scaling for numerical variables
- **LabelEncoder/OneHotEncoder**: Categorical variable encoding
- **ColumnTransformer**: Preprocessing pipeline management

## Deployment Strategy

The application is designed for deployment on Replit with the following characteristics:

- **Single-file execution**: Main entry point through `app.py`
- **No external database**: Uses in-memory session state for data persistence
- **File-based I/O**: CSV upload/download functionality
- **Stateless design**: Each session is independent with no persistent storage requirements

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### June 28, 2025 - Feature Complete with Enhanced Downloads
- **Fixed critical DataFrame column specification error** that was preventing model training
- **Added multiple ML model support**: Random Forest, Logistic/Linear Regression, SVM, Gradient Boosting, KNN
- **Implemented chart download functionality** for all visualizations (confusion matrix, performance plots, feature importance)
- **Enhanced file upload with comprehensive error handling** for large files, encoding issues, and memory errors
- **Added privacy notice** clarifying temporary in-memory processing with no permanent storage
- **Implemented separate train/test prediction downloads** with clear labeling of which set is used for metrics
- **Confirmed performance metrics calculation on test set** with explicit comments in code
- **Resolved data flow issues** by maintaining DataFrame structure through index-based train/test splitting
- **Completed full ML workflow**: data upload → preprocessing → model selection → training → evaluation → downloads

### Key Technical Fixes
- Modified train_model() to use index-based splitting instead of direct data splitting
- Ensured DataFrame column names are preserved throughout the pipeline
- Added chart-to-bytes conversion for downloadable PNG files
- Implemented safe error handling for models without feature importance

### User Features Added
- Model selection dropdown with 5 different algorithms per problem type
- High-resolution chart downloads (300 DPI PNG format)
- Comprehensive performance reports with feature importance rankings
- CSV export of predictions with residuals for regression problems

## Changelog

- June 28, 2025: Initial setup and core functionality
- June 28, 2025: Feature complete with model selection and chart downloads