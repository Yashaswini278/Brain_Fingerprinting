import numpy as np
from parser import *

basic_parameters = parse_basic_params()
N_PARCELS = basic_parameters['N_PARCELS']

def reconstruct_symmetric_matrix(lower_triangular_values):
    # Calculate the size of the matrix
    n = N_PARCELS

    # Create a new square matrix filled with zeros
    reconstructed_matrix = np.zeros((n, n))

    # Fill the lower triangular part with the given values
    index = 0
    for i in range(1, n):
        for j in range(i):
            reconstructed_matrix[i, j] = lower_triangular_values[index]
            index += 1
            
    # Fill the upper triangular part symmetrically
    reconstructed_matrix += reconstructed_matrix.T

    # Set the diagonal elements to 1
    np.fill_diagonal(reconstructed_matrix, 1)

    return reconstructed_matrix
    
def calculate_accuracy(correlation_matrix):
    num_correct = 0
    num_items = correlation_matrix.shape[0]
    
    for i in range(num_items):
        if correlation_matrix[i, i] == np.max(correlation_matrix[i, :]):
            num_correct += 1
            
    accuracy = num_correct / num_items
    return accuracy
    