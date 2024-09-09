# Parameters
data_file = 'halfkp_data.txt'  # Replace with your data file path
batch_size = 64

# Create the dataset and dataloader
dataset = HalfKPDataset(data_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the neural network
model = HalfKPNetwork()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for i, (features, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
