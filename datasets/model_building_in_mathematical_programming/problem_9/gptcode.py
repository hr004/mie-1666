
import json

# Define your function here
def process_data(data):
    # Your code to process data goes here
    pass

# Read data from JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Process data 
output_data = process_data(data)

# Write data to JSON file
with open('output.json', 'w') as file:
    json.dump(output_data, file)
