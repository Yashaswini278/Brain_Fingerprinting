import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting

from models.conv_ae import ConvAutoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Argument parsing for model selection
parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('-model', type=str, help="model to train") 
args = parser.parse_args()

# Load data
data = np.load(r'FC_DATA/fc_rest.npy')
data = data[:, np.newaxis, :, :]  # Reshape to (339, 1, 360, 360)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)

# Create dataset and split into training and validation sets
dataset = TensorDataset(data_tensor, data_tensor)  # Use data as both input and target
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
if args.model == 'conv_ae': 
    model = ConvAutoencoder()

model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20

# Lists to store loss values
train_losses = []
val_losses = []

# Initialize variable to keep track of the best validation loss
best_val_loss = float('inf')
best_model_path = f'./models/trained/{args.model}_best_model.pth'

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_train_loss = 0.0
    
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Compare reconstructed output with the input

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    
    with torch.no_grad():  # No gradient calculation during validation
        for val_data in val_loader:
            val_inputs, _ = val_data
            val_inputs = val_inputs.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_inputs)  # Compare reconstructed output with the input
            running_val_loss += val_loss.item()

    # Calculate average validation loss for the epoch
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Print average losses for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the model if validation loss has decreased
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)  # Save the best model
        print(f'Saved best model with validation loss: {best_val_loss:.4f}')

print('Training complete')

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))  
plt.legend()
plt.grid()

# Save the figure
plt.savefig(f'results/{args.model}/training_validation_loss.png')
plt.close()  
