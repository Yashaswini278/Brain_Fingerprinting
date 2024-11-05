import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdl import *
from parser import *
from utils import *

# Argument parsing for selecting the task and network
parser = argparse.ArgumentParser(description='Functional Connectome Task Selection')
parser.add_argument('-task', type=str, required=True, choices=['motor', 'wm', 'emotion'], help="Task to analyze: motor, wm, or emotion")
parser.add_argument('-network', type=str, required=True, help="Name of the network for analysis (e.g., Visual, Auditory, etc.)")
args = parser.parse_args()

# Basic parameters
basic_parameters = parse_basic_params()
HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']  # This should be consistent regardless of network

# Load the complete functional connectome (FC) data
fc_task= np.load(f'FC_DATA/fc_{args.task}_{args.network}.npy')  # Load the FC data for the specified task

# Extract the network-specific FC data (assuming the network reduces the number of parcels)
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
network_indices = [i for i, net in enumerate(region_info['network']) if net == args.network]

# Average and residuals
fc_task_average = np.mean(fc_task, axis=0)
fc_task_avg = np.repeat(fc_task_average[np.newaxis, ...], fc_task.shape[0], axis=0)
fc_task_residual = fc_task - fc_task_avg

# Apply SDL (Sparse Dictionary Learning)
Y = np.zeros((int(len(network_indices) * (len(network_indices) - 1) / 2), N_SUBJECTS))
for i in range(N_SUBJECTS):
    Y[:, i] = fc_task_residual[i][np.tril_indices(fc_task_residual[i].shape[0], k=-1)]

D, X = k_svd(Y, 15, 12)
sdl_retr = np.dot(D, X).transpose()
DX = np.zeros((N_SUBJECTS, len(network_indices), len(network_indices)))

for i in range(N_SUBJECTS):
    DX[i, :, :] = reconstruct_symmetric_matrix(sdl_retr[i, :], len(network_indices))

fc_task_refined = fc_task_residual - DX

# Plotting the first sample and its reconstruction
first_sample = fc_task[0, :, :]  # Get the first sample (remove channel dimension)
average_sample = fc_task_avg[0, :, :]  # Get the reconstructed output
residual_sample = fc_task_residual[0, :, :]  # Get residual output
refined_sample = fc_task_refined[0, :, :]  # Get refined output

# Create a figure to display the original and reconstructed samples
plt.figure(figsize=(16, 6))

# Plot original sample
plt.subplot(1, 4, 1)
plt.imshow(first_sample, cmap='viridis')
plt.title(f'Original {args.task.capitalize()} Sample ({args.network})')
plt.axis('off')

# Plot reconstructed sample
plt.subplot(1, 4, 2)
plt.imshow(average_sample, cmap='viridis')
plt.title(f'Average {args.task.capitalize()} Sample ({args.network})')
plt.axis('off')

# Plot residual sample
plt.subplot(1, 4, 3)
plt.imshow(residual_sample, cmap='viridis')
plt.title(f'Residual {args.task.capitalize()} Sample ({args.network})')
plt.axis('off')

# Plot refined sample
plt.subplot(1, 4, 4)
plt.imshow(refined_sample, cmap='viridis')
plt.title(f'Refined {args.task.capitalize()} Sample ({args.network})')
plt.axis('off')

# Save the figure
plt.tight_layout()
plt.savefig(f'results/{args.task}_{args.network}_avg_fcs.png')
plt.close()

print(f'Figure saved as "results/{args.task}_{args.network}_avg_fcs.png".')

# Load rest data for comparison
fc_rest = np.load(f'FC_DATA/fc_rest_{args.network}.npy')

# Correlation analysis between task and rest FC before and after processing
corr_task_rest_before_avg = np.corrcoef(
    fc_task.reshape(N_SUBJECTS, -1),  # Reshape to 2D for correlation calculation
    fc_rest.reshape(N_SUBJECTS, -1),  # Reshape to 2D for correlation calculation
    rowvar=True
)[:N_SUBJECTS, N_SUBJECTS:]

corr_task_rest_after_avg = np.corrcoef(
    fc_task_residual.reshape(N_SUBJECTS, -1),  # Reshape to 2D
    fc_rest.reshape(N_SUBJECTS, -1),  # Reshape to 2D
    rowvar=True
)[:N_SUBJECTS, N_SUBJECTS:]

corr_task_rest_after_sdl = np.corrcoef(
    fc_task_refined.reshape(N_SUBJECTS, -1),  # Reshape to 2D
    fc_rest.reshape(N_SUBJECTS, -1),  # Reshape to 2D
    rowvar=True
)[:N_SUBJECTS, N_SUBJECTS:]

# Plot the correlation heatmap
plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.heatmap(corr_task_rest_before_avg, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (Before Average)")

plt.subplot(1, 3, 2)
sns.heatmap(corr_task_rest_after_avg, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After Average)")

plt.subplot(1, 3, 3)
sns.heatmap(corr_task_rest_after_sdl, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After Average+SDL)")

plt.savefig(f'results/{args.task}_avg_corr.png')
plt.close()

print(f'Figure saved as "results/{args.task}_avg_corr.png"')

# Display accuracy results and save them to a text file
accuracy_before = calculate_accuracy(corr_task_rest_before_avg)
accuracy_after = calculate_accuracy(corr_task_rest_after_avg)
accuracy_after_sdl = calculate_accuracy(corr_task_rest_after_sdl)

# Print the results
print(f'{args.task.capitalize()} vs Rest - Accuracy before processing: {accuracy_before}')
print(f'{args.task.capitalize()} vs Rest - Accuracy after Avg: {accuracy_after}')
print(f'{args.task.capitalize()} vs Rest - Accuracy after Avg+SDL: {accuracy_after_sdl}')

# Save the accuracy results to a txt file
with open(f'results/{args.task}_avg_accuracy_results.txt', 'w') as file:
    file.write(f'{args.task.capitalize()} vs Rest - Accuracy before processing: {accuracy_before}\n')
    file.write(f'{args.task.capitalize()} vs Rest - Accuracy after Avg: {accuracy_after}\n')
    file.write(f'{args.task.capitalize()} vs Rest - Accuracy after Avg+SDL: {accuracy_after_sdl}\n')

print(f'Accuracy results saved as "results/{args.task}_avg_accuracy_results.txt".')