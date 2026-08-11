import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Simple MLP for two-number addition
class SimpleAdditionMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Generate training data: addition within specified range
def generate_data(n_samples, min_val, max_val):
    x1 = torch.rand(n_samples, 1) * (max_val - min_val) + min_val
    x2 = torch.rand(n_samples, 1) * (max_val - min_val) + min_val
    inputs = torch.cat([x1, x2], dim=1)
    targets = x1 + x2
    return inputs, targets

# Train the model
print("Training neural network to learn addition on range [-1, 1]...")
model = SimpleAdditionMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training on range [-1, 1]
train_x, train_y = generate_data(1000, -1, 1)

# Training loop
for epoch in range(2000):
    optimizer.zero_grad()
    pred = model(train_x)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("Training completed!")

# Test on expanding ranges to show extrapolation failure
print("Testing extrapolation capability...")
test_ranges = [1, 5, 10, 25, 50, 100]
errors = []

for max_range in test_ranges:
    test_x, test_y = generate_data(200, -1, max_range)
    with torch.no_grad():
        pred = model(test_x)
        error = torch.mean(torch.abs(pred - test_y)).item()
        errors.append(error)
    print(f"Range [-1, {max_range}]: Mean Absolute Error = {error:.3f}")

# Create the plot with log scale for x-axis to better show the dramatic difference
plt.figure(figsize=(12, 8))

# Add green shaded zone for training range
plt.axhspan(0, errors[0]*2, alpha=0.25, color='green', label='Training Range Performance')
# Add vertical shaded area to highlight the training range on x-axis
plt.axvspan(0.8, 1.2, alpha=0.15, color='lightgreen', label='Training Range [-1,1]')

plt.plot(test_ranges, errors, 'ro-', linewidth=3, markersize=12, markerfacecolor='red', markeredgecolor='darkred')
plt.xlabel('Test Range Maximum', fontsize=16, fontweight='bold')
plt.ylabel('Mean Absolute Error', fontsize=16, fontweight='bold')
plt.title('Neural Network Extrapolation Failure in Simple Addition\n(Trained on range [-1,1], tested on expanding ranges)', 
          fontsize=18, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(test_ranges, [str(x) for x in test_ranges], fontsize=14)
plt.yticks(fontsize=14)

# Add failure case examples at specific points using actual model predictions
test_examples = [
    (torch.tensor([[5.0, 3.0]]), 10, errors[2]),
    (torch.tensor([[25.0, 20.0]]), 50, errors[4])
]

failure_annotations = []
with torch.no_grad():
    for test_input, x_pos, y_pos in test_examples:
        pred = model(test_input)
        true_sum = test_input.sum(dim=1).item()
        pred_val = pred.item()
        text = f"{test_input[0,0]:.0f}+{test_input[0,1]:.0f}→{pred_val:.1f}"
        failure_annotations.append((x_pos, y_pos, text))

for x, y, text in failure_annotations:
    plt.annotate(text, xy=(x, y), xytext=(x*0.7, y*1.3),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                fontsize=12, fontweight='bold', color='darkred',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='darkred', alpha=0.9))

# Add legend for the green shaded area over the actual green zone
plt.text(0.08, 0.7, 'Green zone:\nTraining data range [-1, 1]', 
         transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8),
         horizontalalignment='left', verticalalignment='center')

plt.tight_layout()
plt.savefig('3-llms-assets/neural_network_extrapolation_failure.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('3-llms-assets/neural_network_extrapolation_failure.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"\nPlot saved to: 3-llms-assets/neural_network_extrapolation_failure.png")
print(f"Plot also saved as PDF: 3-llms-assets/neural_network_extrapolation_failure.pdf")

# Show some example predictions to illustrate the failure
print("\nExample predictions to illustrate the failure:")
print("=" * 50)

# Test on training range [-1,1]
test_small = torch.tensor([[-0.3, 0.7], [0.1, -0.9], [0.5, -0.4]])
with torch.no_grad():
    pred_small = model(test_small)
    true_small = test_small.sum(dim=1, keepdim=True)

print("Training range [-1,1] - Good performance:")
for i in range(len(test_small)):
    print(f"  {test_small[i,0]:.1f} + {test_small[i,1]:.1f} = {true_small[i,0]:.1f} | Predicted: {pred_small[i,0]:.3f}")

# Test on larger range [-1,100] 
test_large = torch.tensor([[25.0, 50.0], [10.0, 80.0], [30.0, 40.0]])
with torch.no_grad():
    pred_large = model(test_large)
    true_large = test_large.sum(dim=1, keepdim=True)

print("\nLarge range [-1,100] - Catastrophic extrapolation failure:")
for i in range(len(test_large)):
    print(f"  {test_large[i,0]:.0f} + {test_large[i,1]:.0f} = {true_large[i,0]:.0f} | Predicted: {pred_large[i,0]:.1f}")

print("\nThis demonstrates the fundamental limitation of neural networks:")
print("They cannot extrapolate beyond their training distribution, even for simple arithmetic!") 