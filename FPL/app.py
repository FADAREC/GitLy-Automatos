import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to fetch FPL data
def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch FPL data")

# Function to fetch fixture difficulty from FPL
def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception("Failed to fetch FPL fixtures")

# Feature Engineering for Player Data
def prepare_player_data(fpl_data):
    players = pd.DataFrame(fpl_data['elements'])
    teams = pd.DataFrame(fpl_data['teams'])
    positions = pd.DataFrame(fpl_data['element_types'])

    # Map team and position data to players
    players['team_name'] = players['team'].map(teams.set_index('id')['name'])
    players['position'] = players['element_type'].map(positions.set_index('id')['singular_name'])

    # Select useful columns for modeling
    player_features = players[[
        'id', 'web_name', 'team_name', 'position', 'now_cost', 'minutes',
        'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
        'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
        'threat', 'ict_index', 'total_points', 'form', 'selected_by_percent'
    ]]

    # Convert percentage strings to floats
    player_features['selected_by_percent'] = player_features['selected_by_percent'].astype(float)
    
    return player_features

# Merge fixtures data for fixture difficulty calculation
def add_fixture_difficulty(player_data, fixtures):
    fixtures['is_home'] = fixtures['is_home'].astype(int)
    fixtures['difficulty'] = fixtures['difficulty'].astype(int)

    # Aggregate fixture difficulty by team
    team_difficulty = fixtures.groupby('team_h')['difficulty'].mean().reset_index()
    team_difficulty.columns = ['team', 'avg_fixture_difficulty']

    # Map fixture difficulty to players
    player_data = player_data.merge(
        team_difficulty, left_on='team_name', right_on='team', how='left'
    )

    return player_data

# Build a Machine Learning Model
def build_model(player_data):
    # Define features and target variable
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index', 'avg_fixture_difficulty'
    ]]
    target = player_data['total_points']

    # Handle missing values
    features.fillna(0, inplace=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    print(f"Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    return model

# Predict top performers for the next game week
def predict_top_performers(model, player_data):
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index', 'avg_fixture_difficulty'
    ]]

    # Handle missing values
    features.fillna(0, inplace=True)

    player_data['predicted_points'] = model.predict(features)

    # Sort by predicted points
    top_performers = player_data.sort_values(by='predicted_points', ascending=False)
    return top_performers[['web_name', 'team_name', 'position', 'predicted_points']].head(20)

# Main Function
def main():
    fpl_data = fetch_fpl_data()
    fixtures = fetch_fixtures()

    player_data = prepare_player_data(fpl_data)
    player_data = add_fixture_difficulty(player_data, fixtures)

    model = build_model(player_data)
    top_performers = predict_top_performers(model, player_data)

    print("Top Predicted Performers for Next GW:\n")
    print(top_performers)

if __name__ == "__main__":
    main()
