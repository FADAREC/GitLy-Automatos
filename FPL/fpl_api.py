from flask import Flask, request, jsonify
import requests
import requests_cache
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import psutil
import os
import time

# Enable caching
requests_cache.install_cache('fpl_cache', expire_after=3600)  # Cache expires after 1 hour

app = Flask(__name__)

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
    
    players = players[players['minutes'] > 0]  # Filter players who have played at least 1 minute
    players['recent_form'] = players['id'].apply(calculate_recent_form)  # Calculate recent form
    players = players.fillna(0)  # Replace missing values
    
    return players

def add_fixture_difficulty(player_data, fixtures, teams):
    if 'team_h' in fixtures.columns and 'team_h_difficulty' in fixtures.columns:
        team_difficulty = fixtures.groupby('team_h')['team_h_difficulty'].mean().reset_index()
        team_mapping = teams.set_index('id')['name']
        team_difficulty['team_name'] = team_difficulty['team_h'].map(team_mapping)
        player_data = player_data.merge(team_difficulty[['team_name', 'team_h_difficulty']], on='team_name', how='left')
        player_data['team_h_difficulty'].fillna(player_data['team_h_difficulty'].mean(), inplace=True)
    return player_data

def build_and_train_model(player_data):
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index', 'recent_form'
    ]]
    target = player_data['total_points']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, np.sqrt(mean_squared_error(y_test, y_pred))

def predict_top_performers(model, player_data):
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index', 'recent_form'
    ]]
    player_data['predicted_points'] = model.predict(features)
    return player_data.sort_values(by='predicted_points', ascending=False).head(20)

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        fpl_data = fetch_fpl_data()
        fixtures = fetch_fixtures()
        teams = pd.DataFrame(fpl_data['teams'])
        player_data = prepare_player_data(fpl_data)
        player_data = add_fixture_difficulty(player_data, fixtures, teams)
        model, rmse = build_and_train_model(player_data)
        return jsonify({"message": "Model trained successfully", "rmse": rmse})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    try:
        fpl_data = fetch_fpl_data()
        fixtures = fetch_fixtures()
        teams = pd.DataFrame(fpl_data['teams'])
        player_data = prepare_player_data(fpl_data)
        player_data = add_fixture_difficulty(player_data, fixtures, teams)
        model, _ = build_and_train_model(player_data)
        top_performers = predict_top_performers(model, player_data)
        return jsonify(top_performers.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
