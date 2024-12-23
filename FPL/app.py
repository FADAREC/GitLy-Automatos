import requests
import requests_cache
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from tabulate import tabulate
import psutil
import os
import time

# Enable caching
requests_cache.install_cache('fpl_cache', expire_after=3600)  # Cache expires after 1 hour

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch FPL data")

def fetch_player_history(player_id, retries=3):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to fetch player history for player {player_id}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching history for player {player_id}: {e}. Attempt {attempt + 1} of {retries}")
            time.sleep(1)
    return None

def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception("Failed to fetch fixtures data")

def calculate_recent_form(player_id, n=5):
    history = fetch_player_history(player_id)
    if history:
        df = pd.DataFrame(history['history']).tail(n)  # Last n gameweeks
        avg_recent_points = df['total_points'].mean()
        return avg_recent_points
    return 0

def prepare_player_data(fpl_data):
    players = pd.DataFrame(fpl_data['elements'])
    teams = pd.DataFrame(fpl_data['teams']).set_index('id')['name'].to_dict()
    positions = pd.DataFrame(fpl_data['element_types']).set_index('id')['singular_name'].to_dict()

    players['team_name'] = players['team'].map(teams)
    players['position'] = players['element_type'].map(positions)
    
    # Filter players who have played at least 1 minute
    players = players[players['minutes'] > 0]
    
    # Calculate recent form for each player (last 5 gameweeks)
    players['recent_form'] = players['id'].apply(calculate_recent_form)

    # Select and prepare model features
    player_features = players[[
        'id', 'web_name', 'team_name', 'position', 'now_cost', 'minutes', 
        'goals_scored', 'assists', 'clean_sheets', 'bps', 'influence', 
        'creativity', 'threat', 'ict_index', 'total_points', 'recent_form'
    ]]

    # Replace missing values and normalize percentages
    player_features = player_features.fillna(0)
    return player_features

def add_fixture_difficulty(player_data, fixtures, teams):
    print("Fixtures DataFrame:")
    print(fixtures.head())  # Print the first few rows of the fixtures DataFrame for debugging

    if 'team_h' in fixtures.columns and 'team_h_difficulty' in fixtures.columns:
        team_difficulty = fixtures.groupby('team_h')['team_h_difficulty'].mean().reset_index()
        team_difficulty.columns = ['team', 'avg_fixture_difficulty']

        # Map difficulty to team names
        team_mapping = teams.set_index('id')['name']
        team_difficulty['team_name'] = team_difficulty['team'].map(team_mapping)
        team_difficulty.drop(columns=['team'], inplace=True)

        # Merge average fixture difficulty to player data
        player_data = player_data.merge(team_difficulty, on='team_name', how='left')
        player_data['avg_fixture_difficulty'].fillna(player_data['avg_fixture_difficulty'].mean(), inplace=True)
    else:
        raise KeyError("Expected columns 'team_h' and 'team_h_difficulty' not found in fixtures data")

    return player_data

def build_and_train_model(player_data):
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index', 
        'recent_form', 'avg_fixture_difficulty'
    ]]
    target = player_data['total_points']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse:.2f}")
    return model

def predict_top_performers(model, player_data):
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'recent_form', 'avg_fixture_difficulty'
    ]]
    player_data['predicted_points'] = model.predict(features)

    # Sort players by predicted points
    top_performers = player_data.sort_values(by='predicted_points', ascending=False)
    return top_performers[['web_name', 'team_name', 'position', 'predicted_points']].head(20)

def print_top_performers(top_performers):
    print("Top Predicted Performers for Next GW:")
    print(tabulate(top_performers, headers="keys", tablefmt="pretty"))

def main():
    print_memory_usage()
    
    # Fetch data
    print("Fetching FPL data...")
    fpl_data = fetch_fpl_data()
    fixtures = fetch_fixtures()
    teams = pd.DataFrame(fpl_data['teams'])

    # Prepare data
    print("Preparing player data...")
    player_data = prepare_player_data(fpl_data)
    player_data = add_fixture_difficulty(player_data, fixtures, teams)

    # Build and train model
    model = build_and_train_model(player_data)

    # Predict top performers
    print("Predicting top performers...")
    top_performers = predict_top_performers(model, player_data)
    print_top_performers(top_performers)

if __name__ == "__main__":
    main()