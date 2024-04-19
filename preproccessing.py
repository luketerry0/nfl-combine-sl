import pandas as pd
import os




if __name__ == "__main__":
    # append the CSV files
    dfs = []
    for file in os.listdir("./data"):
        if file[0] == '2':
            df = pd.read_csv("./data/" + file)
            
            dfs.append(df)
    df_appended = pd.concat(dfs)

    # safe final csv file
    #df_appended.to_csv("./data/full_data.csv")

    # do some other processing...
    # encode the position
    df_encoded = pd.get_dummies(df_appended, columns=['Pos'])

    # create a new boolean column of if they were drafted
    df_encoded['Drafted'] = pd.isnull(df_encoded['Pick'])

    # replace the height with a number of inches
    def height_to_inches(str):
        try:
            nums = str.split("-")
            return int(nums[0])*12 + int(nums[1])
        except:
            try:
                return int(str)
            except:
                return 0
    df_encoded['Ht'] = df_encoded['Ht'].apply(height_to_inches)

    # replace non-participation with zero...
    df_encoded = df_encoded.fillna(0.0)

    #drop index column, year, player, draft pick info
    df_encoded = df_encoded.drop(columns=['Year', 'Player', 'Round', 'Pick'])

    # convert booleans to 1s or 0s
    df_encoded = 1*df_encoded
    print(df_encoded.shape)
    #df_encoded.to_csv("./data/clean_full_data.csv")


