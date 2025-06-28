import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from ml_utils import (
    detect_problem_type, 
    handle_missing_values, 
    train_model, 
    evaluate_model,
    get_feature_importance
)
from visualization_utils import (
    create_confusion_matrix,
    create_feature_importance_plot,
    create_performance_plots,
    create_data_overview_plots
)

# Configure page
st.set_page_config(
    page_title="ML CSV Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'step' not in st.session_state:
    st.session_state.step = 1

def reset_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        if key not in ['step']:
            del st.session_state[key]
    st.session_state.data = None
    st.session_state.processed_data = None
    st.session_state.model = None
    st.session_state.predictions = None

def download_button(data, filename, label):
    """Create a download button for various file types"""
    if isinstance(data, str):
        # Text data
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>'
    else:
        # Binary data (images)
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{label}</a>'
    
    st.markdown(href, unsafe_allow_html=True)

# Main title
st.title("📊 CSV Machine Learning Analyzer")
st.markdown("Upload your CSV file and build machine learning models with automated analysis and visualizations!")

# Sidebar for navigation
st.sidebar.title("Navigation")
steps = [
    "1. Upload Data",
    "2. Data Preview & Missing Values",
    "3. Feature Selection",
    "4. Model Training",
    "5. Results & Downloads"
]

# Display current step
for i, step_name in enumerate(steps, 1):
    if i == st.session_state.step:
        st.sidebar.markdown(f"**➤ {step_name}**")
    elif i < st.session_state.step:
        st.sidebar.markdown(f"✅ {step_name}")
    else:
        st.sidebar.markdown(f"⭕ {step_name}")

# Reset button
if st.sidebar.button("🔄 Start Over"):
    reset_state()
    st.session_state.step = 1
    st.rerun()

# Step 1: Upload Data
if st.session_state.step == 1:
    st.header("Step 1: Upload Your CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Maximum file size: 50MB"
    )
    
    if uploaded_file is not None:
        # Check file size (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File size exceeds 50MB limit. Please upload a smaller file.")
        else:
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    st.error("Could not read the CSV file. Please check the file format and encoding.")
                else:
                    st.session_state.data = df
                    st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                    
                    # Show basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
                    if st.button("Continue to Data Preview ➡️"):
                        st.session_state.step = 2
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

# Step 2: Data Preview & Missing Values
elif st.session_state.step == 2:
    if st.session_state.data is None:
        st.error("No data found. Please go back to Step 1 and upload a CSV file.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.header("Step 2: Data Preview & Missing Value Handling")
        
        df = st.session_state.data
        
        # Data overview
        st.subheader("📋 Data Overview")
        tab1, tab2, tab3 = st.tabs(["Preview", "Summary", "Missing Values"])
        
        with tab1:
            st.dataframe(df.head(100), use_container_width=True)
        
        with tab2:
            st.write("**Data Types:**")
            st.dataframe(pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            }))
            
            # Numeric columns description
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**Numeric Columns Statistics:**")
                st.dataframe(df[numeric_cols].describe())
        
        with tab3:
            missing_data = df.isnull().sum()
            missing_cols = missing_data[missing_data > 0]
            
            if len(missing_cols) == 0:
                st.success("🎉 No missing values found in your data!")
            else:
                st.warning(f"Found missing values in {len(missing_cols)} columns:")
                
                # Missing value handling
                st.subheader("🔧 Configure Missing Value Handling")
                missing_strategies = {}
                
                for col in missing_cols.index:
                    col_type = df[col].dtype
                    st.write(f"**{col}** ({col_type}) - Missing: {missing_cols[col]} values")
                    
                    if pd.api.types.is_numeric_dtype(col_type):
                        options = ["Drop rows", "Fill with mean", "Fill with median", "Forward fill"]
                        default_idx = 1  # mean
                    else:
                        options = ["Drop rows", "Fill with mode", "Forward fill"]
                        default_idx = 1  # mode
                    
                    strategy = st.selectbox(
                        f"Strategy for {col}:",
                        options,
                        index=default_idx,
                        key=f"strategy_{col}"
                    )
                    missing_strategies[col] = strategy
                
                # Apply missing value handling
                if st.button("Apply Missing Value Handling"):
                    try:
                        processed_df = handle_missing_values(df, missing_strategies)
                        st.session_state.processed_data = processed_df
                        st.success("✅ Missing values handled successfully!")
                        
                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Rows", df.shape[0])
                        with col2:
                            st.metric("Processed Rows", processed_df.shape[0])
                        
                    except Exception as e:
                        st.error(f"Error handling missing values: {str(e)}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Upload"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else df
            if len(missing_cols) == 0 or st.session_state.processed_data is not None:
                if st.button("Continue to Feature Selection ➡️"):
                    if st.session_state.processed_data is None:
                        st.session_state.processed_data = df
                    st.session_state.step = 3
                    st.rerun()

# Step 3: Feature Selection
elif st.session_state.step == 3:
    if st.session_state.processed_data is None:
        st.error("No processed data found. Please go back and complete the previous steps.")
        if st.button("⬅️ Back to Data Preview"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.header("Step 3: Feature Selection")
        
        df = st.session_state.processed_data
        
        st.subheader("🎯 Select Target Variable and Features")
        
        # Target variable selection
        st.write("**Select Target Variable (what you want to predict):**")
        target_col = st.selectbox(
            "Target Column:",
            df.columns,
            help="Choose the column you want to predict"
        )
        
        if target_col:
            # Detect problem type
            problem_type = detect_problem_type(df[target_col])
            
            if problem_type == "classification":
                st.info(f"🔍 **Detected Problem Type:** Classification")
                st.write(f"Unique values in target: {df[target_col].nunique()}")
                if df[target_col].nunique() <= 10:
                    st.write("Target value distribution:")
                    st.bar_chart(df[target_col].value_counts())
            else:
                st.info(f"🔍 **Detected Problem Type:** Regression")
                st.write("Target variable distribution:")
                st.histogram_chart(df[target_col].dropna())
            
            # Feature selection
            st.write("**Select Feature Variables (predictors):**")
            available_features = [col for col in df.columns if col != target_col]
            
            selected_features = st.multiselect(
                "Feature Columns:",
                available_features,
                default=available_features,
                help="Choose the columns to use as predictors"
            )
            
            if selected_features:
                st.write(f"Selected {len(selected_features)} features")
                
                # Show feature info
                feature_info = pd.DataFrame({
                    'Feature': selected_features,
                    'Data Type': [df[col].dtype for col in selected_features],
                    'Non-Null Count': [df[col].count() for col in selected_features],
                    'Unique Values': [df[col].nunique() for col in selected_features]
                })
                st.dataframe(feature_info, use_container_width=True)
                
                # Test size selection
                st.subheader("⚙️ Model Configuration")
                test_size = st.slider(
                    "Test Set Size (percentage):",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5,
                    help="Percentage of data to use for testing"
                ) / 100
                
                # Store selections in session state
                st.session_state.target_col = target_col
                st.session_state.selected_features = selected_features
                st.session_state.problem_type = problem_type
                st.session_state.test_size = test_size
                
                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⬅️ Back to Data Preview"):
                        st.session_state.step = 2
                        st.rerun()
                with col2:
                    if st.button("Train Model ➡️"):
                        st.session_state.step = 4
                        st.rerun()

# Step 4: Model Training
elif st.session_state.step == 4:
    if (st.session_state.processed_data is None or 
        'target_col' not in st.session_state or 
        'selected_features' not in st.session_state):
        st.error("Missing required data. Please go back and complete the previous steps.")
        if st.button("⬅️ Back to Feature Selection"):
            st.session_state.step = 3
            st.rerun()
    else:
        st.header("Step 4: Model Training & Evaluation")
        
        df = st.session_state.processed_data
        target_col = st.session_state.target_col
        selected_features = st.session_state.selected_features
        problem_type = st.session_state.problem_type
        test_size = st.session_state.test_size
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Train model
                    model, X_train, X_test, y_train, y_test, preprocessor = train_model(
                        df, target_col, selected_features, problem_type, test_size
                    )
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Store results
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.preprocessor = preprocessor
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Evaluate model
                    metrics = evaluate_model(y_test, y_pred, problem_type)
                    st.session_state.metrics = metrics
                    
                    # Display results
                    st.subheader("📊 Model Performance")
                    
                    if problem_type == "classification":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                        
                        # Confusion matrix
                        st.subheader("🔥 Confusion Matrix")
                        cm_fig = create_confusion_matrix(y_test, y_pred)
                        st.pyplot(cm_fig)
                        
                    else:  # regression
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{metrics['r2']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['rmse']:.3f}")
                        with col3:
                            st.metric("MAE", f"{metrics['mae']:.3f}")
                        
                        # Actual vs Predicted plot
                        st.subheader("📈 Actual vs Predicted")
                        perf_fig = create_performance_plots(y_test, y_pred, problem_type)
                        st.pyplot(perf_fig)
                    
                    # Feature importance
                    st.subheader("🎯 Feature Importance")
                    importance_scores = get_feature_importance(model, selected_features, preprocessor)
                    importance_fig = create_feature_importance_plot(importance_scores)
                    st.pyplot(importance_fig)
                    
                    # Navigation
                    if st.button("View Results & Downloads ➡️"):
                        st.session_state.step = 5
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.error("Please check your data and feature selection.")
        
        # Navigation buttons
        if st.button("⬅️ Back to Feature Selection"):
            st.session_state.step = 3
            st.rerun()

# Step 5: Results & Downloads
elif st.session_state.step == 5:
    if st.session_state.model is None:
        st.error("No trained model found. Please go back and train a model first.")
        if st.button("⬅️ Back to Model Training"):
            st.session_state.step = 4
            st.rerun()
    else:
        st.header("Step 5: Results & Downloads")
        
        # Display final results
        st.subheader("📋 Model Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Problem Type:** {st.session_state.problem_type.title()}")
            st.write(f"**Target Variable:** {st.session_state.target_col}")
            st.write(f"**Number of Features:** {len(st.session_state.selected_features)}")
        with col2:
            st.write(f"**Training Set Size:** {len(st.session_state.X_train)} samples")
            st.write(f"**Test Set Size:** {len(st.session_state.X_test)} samples")
            st.write(f"**Test Set Ratio:** {st.session_state.test_size:.1%}")
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        metrics = st.session_state.metrics
        
        if st.session_state.problem_type == "classification":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col3:
                st.metric("MAE", f"{metrics['mae']:.4f}")
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        # Generate all plots
        if st.session_state.problem_type == "classification":
            cm_fig = create_confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            st.pyplot(cm_fig)
        else:
            perf_fig = create_performance_plots(
                st.session_state.y_test, 
                st.session_state.y_pred, 
                st.session_state.problem_type
            )
            st.pyplot(perf_fig)
        
        # Feature importance
        importance_scores = get_feature_importance(
            st.session_state.model, 
            st.session_state.selected_features,
            st.session_state.preprocessor
        )
        importance_fig = create_feature_importance_plot(importance_scores)
        st.pyplot(importance_fig)
        
        # Downloads section
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Performance Report**")
            
            # Generate text report
            report = f"""Machine Learning Model Report
=====================================

Dataset Information:
- Total samples: {len(st.session_state.processed_data)}
- Features: {len(st.session_state.selected_features)}
- Target variable: {st.session_state.target_col}
- Problem type: {st.session_state.problem_type}

Model Configuration:
- Train/Test split: {int((1-st.session_state.test_size)*100)}/{int(st.session_state.test_size*100)}
- Training samples: {len(st.session_state.X_train)}
- Test samples: {len(st.session_state.X_test)}

Performance Metrics:
"""
            
            for metric, value in metrics.items():
                report += f"- {metric.replace('_', ' ').title()}: {value:.4f}\n"
            
            report += f"""
Selected Features:
{chr(10).join([f"- {feature}" for feature in st.session_state.selected_features])}

Feature Importance (Top 10):
"""
            
            # Add top 10 feature importance
            sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_importance[:10]:
                report += f"- {feature}: {importance:.4f}\n"
            
            # Download button for report
            st.download_button(
                label="📄 Download Performance Report",
                data=report,
                file_name=f"ml_report_{st.session_state.target_col}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.write("**📊 Predictions CSV**")
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'Actual': st.session_state.y_test,
                'Predicted': st.session_state.y_pred
            })
            
            if st.session_state.problem_type == "regression":
                predictions_df['Residual'] = predictions_df['Actual'] - predictions_df['Predicted']
                predictions_df['Absolute_Error'] = abs(predictions_df['Residual'])
            
            # Download button for predictions
            csv_buffer = io.StringIO()
            predictions_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📊 Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name=f"predictions_{st.session_state.target_col}.csv",
                mime="text/csv"
            )
        
        # Navigation
        st.subheader("🔄 Next Steps")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Model Training"):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("🔄 Start New Analysis"):
                reset_state()
                st.session_state.step = 1
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        Built with Streamlit • Upload CSV → Select Features → Train Model → Download Results
    </div>
    """, 
    unsafe_allow_html=True
)
