import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging  # Import logging
from models.conv_ae import ConvAutoencoder
from sdl import *
from parser import *
from utils import *

# Argument parsing for selecting the task
parser = argparse.ArgumentParser(description='Functional Connectome Task Selection')
parser.add_argument('-task', type=str, required=True, choices=['motor', 'wm', 'emotion'], help="Task to analyze: motor, wm, or emotion")
args = parser.parse_args()

# Set up logging
logging.basicConfig(filename=f'accuracy_log_{args.task}.txt', level=logging.INFO, format='%(message)s', filemode='a')

# Basic parameters
basic_parameters = parse_basic_params()
HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']

# Load the trained model
model = ConvAutoencoder()
model.load_state_dict(torch.load('./models/trained/conv_ae_best_model.pth'))
model.eval()

# Select the appropriate functional connectome (FC) data based on the task
if args.task == 'motor':
    fc_task = np.load('FC_DATA/fc_motor.npy')
elif args.task == 'wm':
    fc_task = np.load('FC_DATA/fc_wm.npy')
elif args.task == 'emotion':
    fc_task = np.load('FC_DATA/fc_emotion.npy')

fc_task = fc_task[:, np.newaxis, :, :]
fc_task_tensor = torch.tensor(fc_task, dtype=torch.float32)

# Perform reconstruction
with torch.no_grad():
    fc_task_reconstr = model(fc_task_tensor)

# Calculate the residuals
fc_task_residual = fc_task_tensor.squeeze(1) - fc_task_reconstr.squeeze(1)

# Load rest data for comparison
fc_rest = np.load('FC_DATA/fc_rest.npy')
fc_rest_tensor = torch.tensor(fc_rest, dtype=torch.float32)

# Correlation analysis function
def calculate_correlation(fc_task_data, fc_rest_data):
    return np.corrcoef(fc_task_data.view(N_SUBJECTS, -1).numpy(), fc_rest_data.view(N_SUBJECTS, -1).numpy(), rowvar=True)[:N_SUBJECTS, N_SUBJECTS:]

# Initialize arrays for storing accuracies
accuracy_matrix = np.zeros((14, 14))  # For K=2 to 15, and L ranging from K to 15

# Loop over K and L, apply SDL, and compute accuracies
for K in range(2, 16):
    for L in range(2, K + 1):
        # Apply SDL
        print(f'K= {K}, L = {L}')
        Y = np.zeros((int(N_PARCELS * (N_PARCELS - 1) / 2), N_SUBJECTS))
        for i in range(N_SUBJECTS):
            Y[:, i] = fc_task_residual[i][np.tril_indices(fc_task_residual[i].shape[0], k=-1)]

        D, X = k_svd(Y, K, L)
        sdl_retr = np.dot(D, X).transpose()
        DX = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

        for i in range(N_SUBJECTS):
            DX[i, :, :] = reconstruct_symmetric_matrix(sdl_retr[i, :])

        fc_task_refined = fc_task_residual - DX

        # Correlation analysis between task and rest FC after SDL
        corr_task_rest_after_sdl = calculate_correlation(fc_task_refined, fc_rest_tensor)

        # Calculate accuracy after SDL
        accuracy_after_sdl = calculate_accuracy(corr_task_rest_after_sdl) * 100  # Convert to percentage
        print(accuracy_after_sdl)

        # Log the accuracy in the specified format
        logging.info(f'K= {K}, L = {L} {accuracy_after_sdl}')

        # Store the accuracy in the matrix
        accuracy_matrix[K - 2, L - 2] = accuracy_after_sdl
