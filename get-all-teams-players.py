import pandas as pd
import json

def extract_unique_players(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Iterate over each row to extract players and associate them with their teams
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Extract home and away players
        home_players = {row[f'home_{i}'] for i in range(5)}
        away_players = {row[f'away_{i}'] for i in range(5)}
        
        # Update dictionary with unique players for each team
        if home_team not in team_players:
            team_players[home_team] = set()
        if away_team not in team_players:
            team_players[away_team] = set()
        
        team_players[home_team].update(home_players)
        team_players[away_team].update(away_players)

# Dictionary to store unique players per team
team_players = {}
for year in range(2007, 2016):
    csv_filename = f'input-dataset/matchups-{year}.csv'
    extract_unique_players(csv_filename)
    
# Save the dictionary to a JSON file
with open('./input-dataset/team-players.json', 'w') as f:
    team_players = {team: sorted(list(players)) for team, players in team_players.items()}
    json.dump(team_players, f)
