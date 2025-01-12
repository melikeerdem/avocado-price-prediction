# 🥑 Avocado Price Prediction Project
This project aims to predict avocado prices using various machine learning models based on historical data. The project includes different approaches such as Linear Regression, Decision Trees, and Support Vector Regression (SVR).
## 📁 Project Structure
```
.
├── datacube.py
├── decision_tree.py
├── linear_regression.py
├── svr.py
└── avocado.csv (required but not included)
```
## ✨ Features
- 🔍 Data preprocessing and feature engineering
- 🤖 Multiple machine learning models:
  - 📈 Linear Regression with polynomial features
  - 🌳 Decision Tree Regression with GridSearchCV
  - 🎯 Support Vector Regression (SVR)
- 📊 Data cube creation for multidimensional analysis
- 📉 Comprehensive visualization of results
- 📌 Model performance metrics (MAE, MSE, RMSE, R², Adjusted R²)
## 🛠️ Prerequisites
- 🐍 Python 3.x
- 🐼 pandas
- 🔢 numpy
- 🧮 scikit-learn
- 📊 matplotlib
- 🎨 seaborn
## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/avocado-price-prediction.git
cd avocado-price-prediction
```
2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
3. Download the avocado.csv dataset and place it in the project root directory.
## 🚀 Usage
### 📊 Data Cube Creation
Run the following to create a multidimensional analysis of the data:
```bash
python datacube.py
```
### 🔬 Model Training and Evaluation
Linear Regression:
```bash
python linear_regression.py
```
Decision Tree:
```bash
python decision_tree.py
```
Support Vector Regression:
```bash
python svr.py
```
## 🤖 Models Overview
### 📈 Linear Regression
- Implements polynomial features
- Uses RobustScaler for feature scaling
- Includes correlation analysis and feature selection
### 🌳 Decision Tree
- Implements GridSearchCV for hyperparameter tuning
- Includes visualization of the tree structure
- Features cross-validation
### 🎯 SVR (Support Vector Regression)
- Uses RBF kernel
- Implements StandardScaler for feature scaling
- Includes error distribution analysis
## 📝 Features Used in Models
- 📊 Total Volume
- 🔢 PLU Codes (4046, 4225, 4770)
- 🛍️ Bag Information (Total, Small, Large)
- 🏷️ Type (Conventional/Organic)
- 📅 Seasonal Features (month_sin, month_cos)
- 🌎 Region (one-hot encoded)
- 📈 Price Momentum
- 🔄 Various calculated ratios
## 📊 Output
Each model script generates:
- 📈 Performance metrics (MAE, MSE, RMSE, R², Adjusted R²)
- 📊 Visualization of actual vs predicted prices
- 📑 Model-specific analytics and visualizations
## 🤝 Contributing
1. 🔱 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. ✍️ Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 🚀 Push to the branch (`git push origin feature/AmazingFeature`)
5. 💫 Open a Pull Request
