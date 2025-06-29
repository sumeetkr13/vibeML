# ML CSV Analyzer

A Streamlit-based machine learning application that provides automated CSV data analysis and model training capabilities. Upload your CSV file and build machine learning models with automated analysis and visualizations!

## 🚀 Features

- **Automated Problem Detection**: Automatically detects whether your dataset requires classification or regression modeling
- **Smart Data Preprocessing**: Handles missing values with multiple strategies (mean, median, mode, forward fill, drop rows)
- **Multiple ML Models**: Choose from Random Forest, Logistic/Linear Regression, SVM, Gradient Boosting, and K-Nearest Neighbors
- **Comprehensive Visualizations**: 
  - Confusion matrices for classification
  - Performance plots and residual analysis for regression
  - Feature importance plots
  - Data overview and correlation plots
- **Complete Results Export**: Download performance reports, predictions, and high-resolution charts
- **Privacy-First**: All data processing happens in-memory with no permanent storage

## 🛠️ Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone <your-repo-url>
cd vibeML

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 2: Using uv (faster)

```bash
# Clone the repository
git clone <your-repo-url>
cd vibeML

# Install with uv
uv pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🌐 Deployment

### Streamlit Cloud

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repository
4. Deploy with `app.py` as the main file

### ngrok (Quick Web Testing)

For immediate web testing with custom URL format:

```bash
./deploy_web.sh
```

This creates a public URL like `https://abc123_SK.ngrok.io` for instant sharing.

### Heroku

1. Create a `Procfile` in the root directory:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy using Heroku CLI:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Railway

1. Connect your GitHub repository to Railway
2. Set the start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### DigitalOcean App Platform

1. Connect your GitHub repository
2. Set the build command: `pip install -r requirements.txt`
3. Set the run command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 📋 Usage

### Step 1: Upload Data
- Upload a CSV file (max 200MB)
- Supports UTF-8, Latin-1, and CP1252 encoding
- View basic dataset information

### Step 2: Data Preview & Missing Values
- Explore your data with interactive tabs
- Configure missing value handling strategies
- View data types and statistics

### Step 3: Feature & Label Selection
- Select your target variable (what you want to predict)
- Choose feature columns (predictors)
- Pick your preferred ML algorithm
- Set train/test split ratio

### Step 4: Model Training
- Train your model with automated preprocessing
- View quick performance metrics
- Monitor training progress

### Step 5: Results & Downloads
- View comprehensive performance metrics
- Download performance reports
- Export predictions as CSV
- Save high-resolution charts

## 🎯 Supported Problem Types

### Classification
- **Models**: Random Forest, Logistic Regression, SVM, Gradient Boosting, K-Nearest Neighbors
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Confusion Matrix, Class Distribution, Accuracy by Class

### Regression
- **Models**: Random Forest, Linear Regression, SVR, Gradient Boosting, K-Nearest Neighbors
- **Metrics**: R² Score, RMSE, MAE
- **Visualizations**: Actual vs Predicted, Residual Plots

## 📊 File Formats

### Input Requirements
- **Format**: CSV files
- **Size**: Maximum 200MB
- **Encoding**: UTF-8, Latin-1, or CP1252
- **Structure**: Tabular data with headers

### Output Files
- **Performance Report**: `.txt` file with detailed metrics
- **Predictions**: `.csv` files with actual vs predicted values
- **Charts**: High-resolution `.png` files (300 DPI)

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **ML Engine**: Scikit-learn with automated model selection
- **Visualization**: Matplotlib/Seaborn
- **Data Processing**: Pandas with comprehensive preprocessing

### Dependencies
- Python 3.11+
- Streamlit 1.46.1+
- Pandas 2.3.0+
- Scikit-learn 1.7.0+
- Matplotlib 3.10.3+
- Seaborn 0.13.2+

## 🛡️ Privacy & Security

- **No Data Storage**: All processing happens in-memory
- **Session-Based**: Each user session is independent
- **No Logging**: No personal data is logged or stored
- **Local Processing**: All computations run locally or on your deployment

## 🔍 Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Ensure file is under 200MB
   - Try re-saving as UTF-8 encoded CSV
   - Check for special characters in column names

2. **Memory Errors**
   - Reduce dataset size by sampling rows
   - Remove unnecessary columns
   - Use a deployment with more RAM

3. **Model Training Fails**
   - Check for missing values in target variable
   - Ensure target variable has valid values
   - Verify feature columns contain numeric/categorical data

### Performance Tips

- For large datasets (>100K rows), consider sampling
- Remove highly correlated features for better performance
- Use smaller test split ratios for small datasets (<1000 rows)

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Support

For questions or issues, please open a GitHub issue or contact the maintainers. 