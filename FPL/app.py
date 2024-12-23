from flask import json
import requests
import requests_cache
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
<<<<<<< HEAD
=======
from concurrent.futures import ThreadPoolExecutor, as_completed
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
from tqdm import tqdm
from tabulate import tabulate
import psutil
import os
import time
<<<<<<< HEAD

# Enable caching
requests_cache.install_cache('fpl_cache', expire_after=3600)  # Cache expires after 1 hour

def print_memory_usage():
=======
from xgboost import XGBRegressor
from bs4 import BeautifulSoup

# Enable caching
requests_cache.install_cache('fpl_cache', expire_after=36000)  # Cache expires automatically after 10 hours

def print_memory_usage():
    """Prints the current memory usage of the process."""
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def fetch_fpl_data():
    """Fetches the FPL data from the official API."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_player_history(player_id, retries=3):
    """Fetches player history data with retries on failure."""
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching history for player {player_id}: {e}. Attempt {attempt + 1} of {retries}")
            time.sleep(1)
    return None

<<<<<<< HEAD
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

=======
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
def fetch_fixtures():
    """Fetches fixture data from the official FPL API."""
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

def calculate_recent_form_parallel(player_ids, n=5):
    """Calculates recent form for players using parallel processing."""
    def fetch_history(player_id):
        history = fetch_player_history(player_id)
        if history:
            df = pd.DataFrame(history['history']).tail(n)
            return df['total_points'].mean() if not df.empty else 0
        return 0

    recent_form = []
    batch_size = 50
    for i in tqdm(range(0, len(player_ids), batch_size)):
        with ThreadPoolExecutor(max_workers=4) as executor:  # Reduced number of threads
            batch_ids = player_ids[i:i + batch_size]
            results = list(executor.map(fetch_history, batch_ids))
            recent_form.extend(results)
    return recent_form

def fetch_understat_data(player_name):
    """Fetches player data from Understat."""
    base_url = f"https://understat.com/player/{player_name}"
    response = requests.get(base_url)
    if response.status_code == 200:
<<<<<<< HEAD
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
=======
        soup = BeautifulSoup(response.content, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            if 'playersData' in script.string:
                json_data = script.string
                start = json_data.index("('") + 2
                end = json_data.index("')")
                json_data = json_data[start:end]
                json_data = json_data.replace("\\", "")
                return json.loads(json_data)
    return None

def fetch_fbref_data(player_name):
    """Fetches player data from FBRef."""
    base_url = f"https://fbref.com/en/players/{player_name}"
    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = pd.read_html(str(soup))
        for table in tables:
            if 'Expected' in table.columns:
                return table
    return None
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57

def prepare_player_data(fpl_data):
    """Prepares player data by merging FPL data with external data."""
    players = pd.DataFrame(fpl_data['elements'])
    teams = pd.DataFrame(fpl_data['teams']).set_index('id')['name'].to_dict()
    positions = pd.DataFrame(fpl_data['element_types']).set_index('id')['singular_name'].to_dict()

    players['team_name'] = players['team'].map(teams)
    players['position'] = players['element_type'].map(positions)
<<<<<<< HEAD
    
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

=======

    # Filter players who have played at least 1 minute
    players = players[players['minutes'] > 0]

    # Calculate recent form
    players['recent_form'] = calculate_recent_form_parallel(players['id'])

    # Additional metrics from external sources
    understat_data = {player: fetch_understat_data(player) for player in players['web_name']}
    fbref_data = {player: fetch_fbref_data(player) for player in players['web_name']}

    # Select and prepare model features
    player_features = players[[
        'id', 'web_name', 'team_name', 'position', 'now_cost', 'minutes',
        'goals_scored', 'assists', 'clean_sheets', 'bps', 'influence',
        'creativity', 'threat', 'ict_index', 'total_points', 'recent_form'
    ]]

    # Merge external metrics
    for player, data in understat_data.items():
        if data:
            player_features.loc[player_features['web_name'] == player, 'xG'] = data.get('xG', 0)
            player_features.loc[player_features['web_name'] == player, 'xA'] = data.get('xA', 0)

    for player, data in fbref_data.items():
        if data is not None:
            player_features.loc[player_features['web_name'] == player, 'xG'] = data.get('xG', 0)
            player_features.loc[player_features['web_name'] == player, 'xA'] = data.get('xA', 0)

>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
    # Replace missing values and normalize percentages
    player_features = player_features.fillna(0)
    return player_features

def add_fixture_difficulty(player_data, fixtures, teams):
<<<<<<< HEAD
=======
    """Adds fixture difficulty to player data."""
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
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
<<<<<<< HEAD
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index', 
        'recent_form', 'avg_fixture_difficulty'
=======
    """Builds and trains the XGBoost model."""
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'recent_form', 'avg_fixture_difficulty', 'xG', 'xA'
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
    ]]
    target = player_data['total_points']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Model training
<<<<<<< HEAD
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    print("Training model...")
=======
    model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
<<<<<<< HEAD
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse:.2f}")
    return model

def predict_top_performers(model, player_data):
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'recent_form', 'avg_fixture_difficulty'
=======
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.2f}")

    return model

def predict_top_performers(model, player_data, n=10):
    """Predicts top performers using the trained model."""
    features = player_data[[
        'now_cost', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'recent_form', 'avg_fixture_difficulty', 'xG', 'xA'
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
    ]]
    player_data['predicted_points'] = model.predict(features)
    return player_data.nlargest(n, 'predicted_points')

<<<<<<< HEAD
    # Sort players by predicted points
    top_performers = player_data.sort_values(by='predicted_points', ascending=False)
    return top_performers[['web_name', 'team_name', 'position', 'predicted_points']].head(20)

def print_top_performers(top_performers):
    print("Top Predicted Performers for Next GW:")
    print(tabulate(top_performers, headers="keys", tablefmt="pretty"))

def main():
    print_memory_usage()
    
=======
def print_top_performers(top_performers):
    """Prints the top performers in a tabulated format."""
    table = top_performers[[
        'web_name', 'team_name', 'position', 'now_cost', 'predicted_points'
    ]]
    print(tabulate(table, headers='keys', tablefmt='pretty'))

def main():
    """Main function to execute the workflow."""
    print_memory_usage()

>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
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
<<<<<<< HEAD
    main()
=======
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
>>>>>>> 85a625fd0d71f976acc4a02cc6d479e152b9ad57
