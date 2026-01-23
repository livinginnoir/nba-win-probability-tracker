from nba_api.stats.static import teams

nba_teams = teams.get_teams()
print(f"Successfully connected! Found {len(nba_teams)} teams.")
