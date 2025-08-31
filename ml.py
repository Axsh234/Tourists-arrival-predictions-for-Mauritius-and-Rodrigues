# ml.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import plotly.graph_objects as go

# 1. Load CSV
df = pd.read_csv("tourism.csv")

# 2. Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# 3. Clean numeric columns
for col in df.columns[1:]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
        .replace("", "0")
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Convert Year to datetime for Prophet
df['Year_dt'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')

# 5. Drop rows with NaN after cleaning
df = df.dropna().reset_index(drop=True)

print("Cleaned Data:")
print(df.head())

# 6. Prepare features and target
X = df[['Year']].values
y_mauritius = df['Mauritius_Arrivals'].values

# 7. Define metric function
def compute_metrics(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # fixed for older scikit-learn
    print(f"{model_name} - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return r2, mae, rmse

# 8. Linear Regression
lr = LinearRegression()
lr.fit(X, y_mauritius)
y_pred_lr = lr.predict(X)
compute_metrics(y_mauritius, y_pred_lr, "Linear Regression (Mauritius)")

# 9. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y_mauritius)
y_pred_rf = rf.predict(X)
compute_metrics(y_mauritius, y_pred_rf, "Random Forest (Mauritius)")

# 10. Prophet
prophet_df = df[['Year_dt', 'Mauritius_Arrivals']].rename(columns={'Year_dt':'ds', 'Mauritius_Arrivals':'y'})
prophet_model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=5, freq='Y')
forecast = prophet_model.predict(future)

# 11. Plot results with Plotly
fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(x=df['Year'], y=y_mauritius, mode='markers+lines', name='Actual', line=dict(color='blue')))

# Linear Regression
fig.add_trace(go.Scatter(x=df['Year'], y=y_pred_lr, mode='lines', name='Linear Regression', line=dict(color='red', dash='dash')))

# Random Forest
fig.add_trace(go.Scatter(x=df['Year'], y=y_pred_rf, mode='lines', name='Random Forest', line=dict(color='green', dash='dot')))

# Prophet
fig.add_trace(go.Scatter(x=forecast['ds'].dt.year, y=forecast['yhat'], mode='lines', name='Prophet', line=dict(color='orange', dash='dashdot')))

fig.update_layout(title='Mauritius Arrivals Forecasting',
                  xaxis_title='Year',
                  yaxis_title='Arrivals',
                  template='plotly_white')

fig.show()
