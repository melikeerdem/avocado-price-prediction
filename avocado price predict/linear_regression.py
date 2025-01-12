import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('avocado.csv')

# Map the "type" column to numerical values (0 for conventional, 1 for organic)
df['type'] = df['type'].map({'conventional': 0, 'organic': 1})

# Convert categorical "region" column to dummy variables
region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
df = pd.concat([df, region_dummies], axis=1)

# Convert "Date" column to datetime format and extract the month
df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month

# Create "Price Momentum" as the lagged average price by region
df['Price_Momentum'] = df.groupby('region')['AveragePrice'].shift(1)

# Avoid division by zero when calculating ratios
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

# Aggregate volume from specific PLU codes
df['Total_PLU_Volume'] = df['4046'] + df['4225'] + df['4770']

# Encode seasonal features using sine and cosine transformations
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Drop the original "region" column as it has been encoded
# and is no longer needed
df.drop('region', axis=1, inplace=True)

# Feature list for the model
features = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 
           'Small_Bags_Ratio', 'Large_Bags_Ratio', 'type', 
           'year', 'month_sin', 'month_cos', 'Volume_per_Bag',
           'Total_PLU_Volume', 'Price_Momentum'] + list(region_dummies.columns)

# Fill missing values
df = df.ffill()

# Compute the correlation matrix
correlation_matrix = df[features].corr()

# Identify pairs of features with high correlation (> 0.8)
high_corr_features = np.where(np.abs(correlation_matrix) > 0.8)
high_corr_features = [(features[x], features[y]) for x, y in zip(*high_corr_features) if x != y and x < y]

# Remove one feature from each highly correlated pair
features_to_remove = set()
for feat1, feat2 in high_corr_features:
    features_to_remove.add(feat2)

# Final feature selection
selected_features = [f for f in features if f not in features_to_remove]

# Prepare the feature matrix (X) and target variable (y)
X = df[selected_features]
y = df['AveragePrice']

# Fill any remaining missing values in the feature matrix
X = X.ffill().bfill().fillna(0)

# Scale the features using RobustScaler to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Transform the features into polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Dictionary to store the results
results = {}

def adjusted_r2(r2, n, p):
    """Calculate adjusted R-squared."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Generate predictions for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(r2, len(X_test), len(selected_features))

results['Linear Regression'] = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'Adjusted R2': adj_r2
}

# Print the evaluation metrics
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    print("------------------------")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Avocado Prices')
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
