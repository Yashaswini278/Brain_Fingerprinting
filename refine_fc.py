import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from models.conv_ae import ConvAutoencoder
from sdl import *
from parser import *
from utils import * 

# Argument parsing for selecting the task
parser = argparse.ArgumentParser(description='Functional Connectome Task Selection')
parser.add_argument('-task', type=str, required=True, choices=['motor', 'wm', 'emotion'], help="Task to analyze: motor, wm, or emotion")
args = parser.parse_args()

# Basic parameters
basic_parameters = parse_basic_params()
HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']

# Load the trained model
model = ConvAutoencoder()
model.load_state_dict(torch.load('./models/trained/conv_ae_best_model.pth'))  # Use torch.load to load model state dict
model.eval()  # Set the model to evaluation mode

# Select the appropriate functional connectome (FC) data based on the task
if args.task == 'motor':
    fc_task = np.load('FC_DATA/fc_motor.npy')
elif args.task == 'wm':
    fc_task = np.load('FC_DATA/fc_wm.npy')
elif args.task == 'emotion':
    fc_task = np.load('FC_DATA/fc_emotion.npy')

fc_task = fc_task[:, np.newaxis, :, :]  # Reshape to (N_SUBJECTS, 1, N_PARCELS, N_PARCELS)
fc_task_tensor = torch.tensor(fc_task, dtype=torch.float32)

# Perform reconstruction
with torch.no_grad():
    fc_task_reconstr = model(fc_task_tensor)

# Calculate the residuals
fc_task_residual = fc_task_tensor.squeeze(1) - fc_task_reconstr.squeeze(1)

# Apply SDL (Sparse Dictionary Learning)
Y = np.zeros((int(N_PARCELS*(N_PARCELS - 1)/2), N_SUBJECTS))
for i in range(N_SUBJECTS):
    Y[:, i] = fc_task_residual[i][np.tril_indices(fc_task_residual[i].shape[0], k=-1)]

D, X = k_svd(Y, 2, 2)
sdl_retr = np.dot(D, X).transpose()
DX = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

for i in range(N_SUBJECTS):
    DX[i, :, :] = reconstruct_symmetric_matrix(sdl_retr[i, :])

fc_task_refined = fc_task_residual - DX

# Plotting the first sample and its reconstruction
first_sample = fc_task[0, 0, :, :]  # Get the first sample (remove channel dimension)
reconstructed_sample = fc_task_reconstr[0, 0, :, :].numpy()  # Get the reconstructed output
residual_sample = fc_task_residual[0, :, :].numpy()  # Get residual output
refined_sample = fc_task_refined[0, :, :].numpy()  # Get refined output

# Create a figure to display the original and reconstructed samples
plt.figure(figsize=(16, 6))

# Plot original sample
plt.subplot(1, 4, 1)
plt.imshow(first_sample, cmap='viridis')
plt.title(f'Original {args.task.capitalize()} Sample')
plt.axis('off')

# Plot reconstructed sample
plt.subplot(1, 4, 2)
plt.imshow(reconstructed_sample, cmap='viridis')
plt.title(f'Reconstructed {args.task.capitalize()} Sample')
plt.axis('off')

# Plot residual sample
plt.subplot(1, 4, 3)
plt.imshow(residual_sample, cmap='viridis')
plt.title(f'Residual {args.task.capitalize()} Sample')
plt.axis('off')

# Plot refined sample
plt.subplot(1, 4, 4)
plt.imshow(refined_sample, cmap='viridis')
plt.title(f'Refined {args.task.capitalize()} Sample')
plt.axis('off')

# Save the figure
plt.tight_layout()
plt.savefig(f'results/{args.task}_fcs.png')
plt.close()

print(f'Figure saved as "results/{args.task}_fcs.png".')

# Load rest data for comparison
fc_rest = np.load('FC_DATA/fc_rest.npy')
fc_rest_tensor = torch.tensor(fc_rest, dtype=torch.float32)

# Correlation analysis between task and rest FC before and after processing
corr_task_rest_before_convae = np.corrcoef(fc_task_tensor.view(N_SUBJECTS, -1).numpy(), fc_rest_tensor.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]
corr_task_rest_after_convae = np.corrcoef(fc_task_residual.view(N_SUBJECTS, -1).numpy(), fc_rest_tensor.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]
corr_task_rest_after_sdl = np.corrcoef(fc_task_refined.view(N_SUBJECTS, -1).numpy(), fc_rest_tensor.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]

# Plot the correlation heatmap
plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.heatmap(corr_task_rest_before_convae, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (Before ConvAE)")

plt.subplot(1, 3, 2)
sns.heatmap(corr_task_rest_after_convae, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After ConvAE)")

plt.subplot(1, 3, 3)
sns.heatmap(corr_task_rest_after_sdl, annot=False, cmap="coolwarm", cbar=True)
plt.title(f"Correlation Matrix - {args.task.capitalize()} vs Rest (After ConvAE+SDL)")

plt.savefig(f'results/{args.task}_corr.png')
plt.close()

print(f'Figure saved as "results/{args.task}_corr.png"')

# Display accuracy results and save them to a text file
accuracy_before = calculate_accuracy(corr_task_rest_before_convae)
accuracy_after_convae = calculate_accuracy(corr_task_rest_after_convae)
accuracy_after_sdl = calculate_accuracy(corr_task_rest_after_sdl)

# Print the results
print(f'{args.task.capitalize()} vs Rest - Accuracy before processing: {accuracy_before}')
print(f'{args.task.capitalize()} vs Rest - Accuracy after ConvAutoEncoder: {accuracy_after_convae}')
print(f'{args.task.capitalize()} vs Rest - Accuracy after ConvAutoEncoder+SDL: {accuracy_after_sdl}')

# Save the accuracy results to a txt file
with open(f'results/{args.task}_accuracy_results.txt', 'w') as file:
    file.write(f'{args.task.capitalize()} vs Rest - Accuracy before processing: {accuracy_before}\n')
    file.write(f'{args.task.capitalize()} vs Rest - Accuracy after ConvAutoEncoder: {accuracy_after_convae}\n')
    file.write(f'{args.task.capitalize()} vs Rest - Accuracy after ConvAutoEncoder+SDL: {accuracy_after_sdl}\n')

print(f'Accuracy results saved as "results/{args.task}_accuracy_results.txt".')