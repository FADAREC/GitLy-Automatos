import pandas as pd
from sklearn.metrics import mean_squared_error

# Step 1: Filter data for a specific gameweek (e.g., GW10)
gameweek = 10  # Choose the past gameweek you want to analyze
gw_data = player_data[player_data['gameweek'] == gameweek]

# Step 2: Split features and target for that GW
features_gw = gw_data[['total_points', 'goals_scored', 'assists', 'clean_sheets', 'saves']].fillna(0)
actual_points = gw_data['actual_points']  # Assuming actual_points column exists

# Step 3: Predict points for that GW
predicted_points = model.predict(features_gw)

# Add predictions and actual points to the DataFrame
gw_data = gw_data.copy()  # To avoid SettingWithCopyWarning
gw_data['predicted_points'] = predicted_points
gw_data['error'] = gw_data['actual_points'] - gw_data['predicted_points']

# Step 4: Analyze overperformers and underperformers
overperformers = gw_data[gw_data['error'] > 0].sort_values(by='error', ascending=False)
underperformers = gw_data[gw_data['error'] < 0].sort_values(by='error')

# Step 5: Print results
print(f"Performance Analysis for Gameweek {gameweek}:\n")

print("Top Overperformers:")
print(overperformers[['web_name', 'team_name', 'position', 'predicted_points', 'actual_points', 'error']].head(10))

print("\nTop Underperformers:")
print(underperformers[['web_name', 'team_name', 'position', 'predicted_points', 'actual_points', 'error']].head(10))

# Step 6: Evaluate Model Accuracy for that GW
rmse = mean_squared_error(actual_points, predicted_points, squared=False)
print(f"\nModel RMSE for Gameweek {gameweek}: {rmse:.2f}")
