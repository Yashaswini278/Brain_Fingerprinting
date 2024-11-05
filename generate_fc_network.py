import os
import argparse
import numpy as np
from hcp_helper_functions import *
from parser import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load timeseries for a specific task.')
parser.add_argument('-task', type=str, help='Name of the task (e.g., "rest", "motor")')
parser.add_argument('-network', type=str, help='Name of the network for functional connectivity')
args = parser.parse_args()

# Load basic parameters
basic_parameters = parse_basic_params()

HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']

subjects = range(N_SUBJECTS)

# Load the regions data
regions = np.load('DATA/hcp_rest/regions.npy', allow_pickle=True)

# Access the data for each region
region_names = regions[:, 0].tolist()  # Extract the first column (names)
network_types = regions[:, 1].tolist()  # Extract the second column (network types)
myelin_values = regions[:, 2].astype(float)  # Convert the third column (myelin) to float

# Create a dictionary to hold the region information
region_info = {
    'name': region_names,
    'network': network_types,
    'myelin': myelin_values
}

# Get the list of unique network names
network_names = np.unique(region_info["network"])

# Check if the specified network is valid
if args.network not in network_names:
    raise ValueError(f"Invalid network name. Available networks: {network_names}")

# Prepare to load fMRI data
fmri_data = []
for subject in subjects:
    # Set directory based on the task name
    task_dir = os.path.join(HCP_DIR, "hcp_rest" if args.task.lower() == "rest" else "hcp_task")

    ts_concat = load_timeseries(subject, name=args.task, dir=task_dir)
    fmri_data.append(ts_concat)

fmri_data_numpy = np.array(fmri_data)

print(f"Loaded fMRI data for '{args.task}'")

# Identify indices for the specified network
network_indices = [i for i, net in enumerate(region_info['network']) if net == args.network]

# Initialize FC matrix
N_PARCELS = len(network_indices)  # Update the number of parcels based on the selected network
fc = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

for sub, ts in enumerate(fmri_data):
    # Select time series only for the specified network
    ts_network = ts[:, network_indices]
    
    # Compute the correlation matrix for the selected regions
    fc[sub] = np.corrcoef(ts_network, rowvar=False)  # Ensure that rows represent regions

print(f"Generated Functional Connectomes for the '{args.network}' network of shape = {fc.shape}")

# Save the functional connectivity array
fc_filename = os.path.join('./FC_DATA', f'fc_{args.task}_{args.network}.npy')  # Define the filename
np.save(fc_filename, fc)  # Save the array
print(f"Saved functional connectivity matrix to '{fc_filename}'")