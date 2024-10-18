import os
import argparse
from hcp_helper_functions import *
from parser import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load timeseries for a specific task.')
parser.add_argument('-task', type=str, help='Name of the task (e.g., "rest", "motor")')
args = parser.parse_args()

# Load basic parameters
basic_parameters = parse_basic_params()

HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']

subjects = range(N_SUBJECTS)

fmri_data = []
for subject in subjects:
    # Set directory based on the task name
    if args.task.lower() == "rest":
        task_dir = os.path.join(HCP_DIR, "hcp_rest")
    else:
        task_dir = os.path.join(HCP_DIR, "hcp_task")

    ts_concat = load_timeseries(subject, name=args.task, dir=task_dir)
    fmri_data.append(ts_concat)

fmri_data_numpy = np.array(fmri_data)

print(f"Loaded fMRI data for '{args.task}'")
print(f"Shape = {fmri_data_numpy.shape}")

fc = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
for sub, ts in enumerate(fmri_data):
  fc[sub] = np.corrcoef(ts)

print(f"Generated Functional Connectomes")

# Save the functional connectivity array
fc_filename = os.path.join('./FC_DATA', f'fc_{args.task}.npy')  # Define the filename
np.save(fc_filename, fc)  # Save the array
print(f"Saved functional connectivity matrix to '{fc_filename}'")