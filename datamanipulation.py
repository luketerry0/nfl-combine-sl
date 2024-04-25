import pandas as pd

# Read the submission and test data CSV files
submission = pd.read_csv('submission.csv')
test_data = pd.read_csv('testwithpick.csv')

# Merge the submission and test data on the 'Index' column
merged_data = pd.merge(submission, test_data, on='Index', suffixes=('_predicted', '_true'))

# Calculate the number of correct predictions
num_correct_predictions = (merged_data['Pick_predicted'] == merged_data['Pick_true']).sum()

# Calculate the total number of predictions
total_predictions = merged_data.shape[0]

# Calculate the percentage of correct predictions
percentage_correct = (num_correct_predictions / total_predictions) * 100

print(f"Percentage of correct predictions: {percentage_correct:.2f}%")
