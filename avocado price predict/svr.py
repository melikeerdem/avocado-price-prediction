import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('avocado.csv')

# Map avocado type to numeric values ('conventional': 0, 'organic': 1)
df['type'] = df['type'].map({'conventional': 0, 'organic': 1})

# One-hot encode the 'region' column
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
df = pd.concat([df, region_dummies], axis=1)

# Convert 'Date' column to datetime format and extract month
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month

# Calculate price momentum for each region (lagged average price)
df['Price_Momentum'] = df.groupby('region')['AveragePrice'].shift(1)

# Prevent division by zero in ratio calculations
df['Volume_per_Bag'] = np.where(
    df['Total Bags'] > 0,
    df['Total Volume'] / df['Total Bags'],
    0
)
df['Small_Bags_Ratio'] = np.where(
    df['Total Bags'] > 0,
    df['Small Bags'] / df['Total Bags'],
    0
)
df['Large_Bags_Ratio'] = np.where(
    df['Total Bags'] > 0,
    df['Large Bags'] / df['Total Bags'],
    0
)

# Sum PLU volumes
df['Total_PLU_Volume'] = df['4046'] + df['4225'] + df['4770']

# Add seasonal features using sine and cosine transformations
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Drop the original 'region' column
df.drop('region', axis=1, inplace=True)

# Define feature set
features = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 
           'Small_Bags_Ratio', 'Large_Bags_Ratio', 'type', 
           'year', 'month_sin', 'month_cos', 'Volume_per_Bag',
           'Total_PLU_Volume', 'Price_Momentum'] + list(region_dummies.columns)

# Fill NaN values
df = df.ffill()

# Perform correlation analysis to identify highly correlated features
correlation_matrix = df[features].corr()

# Identify and remove highly correlated features (> 0.8 correlation)
high_corr_features = np.where(np.abs(correlation_matrix) > 0.8)
high_corr_features = [(features[x], features[y]) for x, y in zip(*high_corr_features) if x != y and x < y]

features_to_remove = set()
for feat1, feat2 in high_corr_features:
    features_to_remove.add(feat2)

selected_features = [f for f in features if f not in features_to_remove]

# Prepare the dataset
X = df[selected_features]
y = df['AveragePrice']

# Fill missing values in feature set
X = X.ffill().bfill().fillna(0)

# Scale the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create the SVR model
model = SVR(
    kernel='rbf',     # Radial basis function kernel
    C=1.0,           # Regularization parameter
    epsilon=0.1,     # Epsilon in the epsilon-SVR model
    gamma='scale'    # Kernel coefficient
)
model.fit(X_train, y_train)

# Define a function to calculate adjusted R-squared
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Predict target values on the test set
y_pred = model.predict(X_test)

# Calculate metrics for model evaluation
results = {}
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(r2, len(X_test), X_test.shape[1])

results['SVR'] = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'Adjusted R2': adj_r2
}

# Print the evaluation metrics
print("\nSVR Model Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Avocado Prices')
plt.show()

# Visualize error distribution
plt.figure(figsize=(10, 6))
error_percentage = (abs(y_test - y_pred) / y_test) * 100
sns.histplot(error_percentage, kde=True)
plt.xlabel('Prediction Error (%)')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.show()

# Visualize evaluation metrics
plt.figure(figsize=(15, 8))
metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'Adjusted R2']
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    values = [results[model][metric] for model in results]
    plt.bar(results.keys(), values)
    plt.title(metric)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
