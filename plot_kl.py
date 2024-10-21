import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

accuracy_matrix = np.zeros((14, 14))  # K=2 to 15, L=2 to 15

# Read the log file
with open('accuracy_log_motor.txt', 'r') as file:
    for line in file:
        # Split the line into parts
        parts = [part.strip(' ,=') for part in line.strip().split() if part.strip(' ,=')]
        
        # Extract K, L, and accuracy
        k = int(parts[1].strip(' =,'))  # Remove spaces, '=', and ','
        l = int(parts[3].strip(' =,'))  # Remove spaces, '=', and ','
        accuracy = float(parts[4])  # Accuracy is at index 6
        
        # Store accuracy in the appropriate location in the matrix
        accuracy_matrix[k - 2, l - 2] = accuracy  # Subtracting 2 to index from 0

# Set up the colormap and normalization
cmap = sns.color_palette("plasma", as_cmap=True)  # You can choose a different colormap if desired
norm = Normalize(vmin=np.min(accuracy_matrix[accuracy_matrix > 0]), vmax=np.max(accuracy_matrix))

# Plot heatmap of accuracies with increased contrast for non-zero values
plt.figure(figsize=(10, 8))
sns.heatmap(accuracy_matrix, annot=True, fmt=".2f", cmap=cmap, norm=norm, 
            xticklabels=range(2, 16), yticklabels=range(2, 16))
plt.xlabel('L')
plt.ylabel('K')
plt.title('Accuracy Heatmap (K=2 to 15, L=2 to K)')
plt.savefig('results/acc_kl_motor.png')  # Save the figure
plt.show()  # Show the plot
