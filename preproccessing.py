import pandas as pd
import os


# append the CSV files
dfs = []
for file in os.listdir("./data"):
    df = pd.read_csv("./data/" + file)
    
    dfs.append(df)
df_appended = pd.concat(dfs)

# safe final csv file
df_appended.to_csv("./data/LR_data.csv")

