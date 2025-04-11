import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("dataset_with_failure_type.csv")

# Extract unique values for Product_ID and Type
unique_product_ids = data['Product_ID'].unique()
unique_types = data['Type'].unique()

# Get the last UDI value
last_udi = data['UDI'].max()

# Define the number of rows to add
num_new_rows = 100  # You can adjust this value as needed

# Generate new rows
new_rows = []
for i in range(num_new_rows):
    # Increment UDI
    udi = last_udi + i + 1
    
    # Randomly select Product_ID and Type
    product_id = np.random.choice(unique_product_ids)
    type_ = np.random.choice(unique_types)
    
    # Generate random values for other features based on observed ranges
    air_temperature = np.round(np.random.uniform(295.0, 304.5), 1)
    process_temperature = np.round(air_temperature + np.random.uniform(8.0, 12.0), 1)
    rotational_speed = np.random.randint(1000, 2500)
    torque = np.round(np.random.uniform(20.0, 70.0), 1)
    tool_wear = np.random.randint(0, 250)
    
    # Simulate failure-related columns (all zeros for simplicity)
    machine_failure = 0
    twf = 0
    hdf = 0
    pwf = 0
    osf = 0
    rnf = 0
    failure_type = 0  # Default to "No Failure"
    
    # Append the new row
    new_row = {
        'UDI': udi,
        'Product_ID': product_id,
        'Type': type_,
        'Air_temperature': air_temperature,
        'Process_temperature': process_temperature,
        'Rotational_speed': rotational_speed,
        'Torque': torque,
        'Tool_wear': tool_wear,
        'Machine_failure': machine_failure,
        'TWF': twf,
        'HDF': hdf,
        'PWF': pwf,
        'OSF': osf,
        'RNF': rnf,
        'Failure_Type': failure_type
    }
    new_rows.append(new_row)

# Convert the list of new rows into a DataFrame
new_data = pd.DataFrame(new_rows)

# Combine the new rows with the original dataset
extended_data = pd.concat([data, new_data], ignore_index=True)

# Save the extended dataset to a new CSV file
extended_data.to_csv("extended_dataset.csv", index=False)

# Print the first few rows of the extended dataset
print(extended_data.tail())