import csv

# Function to convert height from feet-inches format to inches
def height_to_inches(height):
    if '-' in height:
        feet, inches = height.split('-')
        return int(feet) * 12 + int(inches)
    else:
        return int(height)  # Assuming height is already in inches

# Function to convert player position to number
def position_to_number(position):
    if position == "QB":
        return 1
    elif position == "WR":
        return 2
    elif position == "CB":
        return 3
    else:
        return 4

# Read the original CSV file
with open('LR_data.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header
    data = []
    for row in reader:
        # Fill unfilled data points with 0's
        row = [col if col else '0' for col in row]
        # Convert player position to number
        row[3] = position_to_number(row[3])
        # Convert height to inches
        row[4] = height_to_inches(row[4])
        data.append(row)

# Write the updated data to a new CSV file
with open('newLR_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # Write the header
    writer.writerows(data)