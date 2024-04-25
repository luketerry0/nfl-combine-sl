import pandas as pd

# Load the data
allplayers_df = pd.read_csv('allplayers.csv')

# Count the number of players drafted (Pick = 1)
drafted_players_count = allplayers_df['Pick'].sum()

# Total number of players
total_players = len(allplayers_df)

# Calculate the percentage of drafted players
percentage_drafted = (drafted_players_count / total_players) * 100

print("Percentage of drafted players: {:.2f}%".format(percentage_drafted))

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('LR_Data.csv')
#
# # Count the number of players with a pick greater than 0
# drafted_players = df[df['Pick'] > 0]['Pick'].count()
#
# # Total number of players
# total_players = len(df)
#
# # Calculate the percentage of drafted players
# percentage_drafted = (drafted_players / total_players) * 100
#
# print("Percentage of players drafted: {:.2f}%".format(percentage_drafted))

