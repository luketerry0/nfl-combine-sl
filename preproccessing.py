import pandas as pd
import os

# class to let me easily get clean data from the different datasets
class Data():
    def get_data(na_treatment = "zeroed", exclude = [], proportions = [0.9, 0.1]):
        data_path = ""
        if na_treatment == "zeroed":
            data_path = "./data/data_null_values_0.csv"
        elif na_treatment == "averaged":
            data_path = "./data/data_null_values_avg.csv"
        elif na_treatment == "dropped":
            data_path = "data/data_drop_na.csv"
        else:
            raise ValueError("Incorrect Value for Treatment of NA datapoints")
        
        # read the data into a pandas array
        df = pd.read_csv(data_path)
        df = df.drop('Unnamed: 0', axis=1)

        # shuffle the data in place
        df = df.sample(frac=1).reset_index(drop=True)

        datasets = []
        split_points = [round(df.shape[0]*prop) for prop in proportions]
        # split the data into the correct proportions
        prev = 0
        for split_point in split_points:
            curr_df = df.iloc[prev:(split_point + prev)]
            prev = split_point
            datasets.append((curr_df.drop("Drafted", axis=1).to_numpy(), curr_df["Drafted"].to_numpy()))


        return datasets

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

    #drop index column, year, player, draft pick info
    df_encoded = df_encoded.drop(columns=['Year', 'Player', 'Round', 'Pick'])

    # convert booleans to 1s or 0s
    df_encoded = 1*df_encoded
    
    # replace non-participation with zero...
    data_0_for_na = df_encoded.fillna(0.0)
    data_0_for_na.to_csv("./data/data_null_values_0.csv")

    # replace non-participation with average
    data_avg_for_na = df_encoded.fillna(df_encoded.mean())
    data_avg_for_na.to_csv("./data/data_null_values_avg.csv")

    # only keep full participants
    data_full_partic = df_encoded.dropna()
    data_full_partic.to_csv("./data/data_drop_na.csv")

