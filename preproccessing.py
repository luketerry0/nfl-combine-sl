import pandas as pd
import os


# append the CSV files
dfs = []
for file in os.listdir("./data"):
    df = pd.read_csv("./data/" + file)
    
    dfs.append(df)
df_appended = pd.concat(dfs)

# safe final csv file

df_appended.to_csv("./data/full_data.csv")

# do some other processing...
# encode the position
df_encoded = pd.get_dummies(df_appended, columns=['Pos'])

# create a new boolean column of if they were drafted
df_encoded['Drafted'] = pd.isnull(df_encoded['Pick'])

# replace non-participation with zero...
df_encoded = df_encoded.fillna(0.0)

#drop index column, year, player, draft pick info
df_encoded = df_encoded.drop(columns=['Unnamed: 0', 'Year', 'Player', 'Round', 'Pick'])

df_encoded.to_csv("./data/clean_full_data.csv")
