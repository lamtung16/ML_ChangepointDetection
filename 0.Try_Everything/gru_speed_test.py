import torch
import torch.nn as nn
import time

# Define the GRU model
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden

# Generate toy data
input_data = torch.rand(40000).view(-1, 1)
target = torch.tensor([[2.0] * 8])

# Define the model, loss function, and optimizer
model = SimpleGRU(input_size=1, hidden_size=8)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)

# Start training
start_time = time.time()

for epoch in range(1):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, target)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Training time for one epoch: {end_time - start_time} seconds")


