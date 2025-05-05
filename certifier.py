import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn as nn
from interval_propagation import propagate_bounds
from train_model import SimpleMLP
import numpy as np

# Load trained model
model = SimpleMLP()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load one image
dataset = MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
index = 1 
x, y = dataset[index]
x = x.unsqueeze(0)  # Add batch dimension

epsilon = 0.001  # epsilon (perturbation bound)

# Make clean prediction
with torch.no_grad():
    logits = model(x).squeeze()
    pred_class = logits.argmax().item()
    pred_conf = logits[pred_class].item()

print(f"\nClean Prediction: {pred_class} (Confidence: {pred_conf:.4f}) | True Label: {y} | Epsilon: {epsilon}")
print(f"Prediction {'CORRECT' if pred_class == y else 'WRONG'}")

# Run interval bound propagation
l_bounds, u_bounds = propagate_bounds(model, x[0], epsilon)

print("\n Output Logit Bounds:")
margin_info = []
for i in range(len(l_bounds)):
    range_width = u_bounds[i] - l_bounds[i]
    print(f"Class {i}: Lower = {l_bounds[i]:.4f}, Upper = {u_bounds[i]:.4f} | Range Width: {range_width:.4f}")
    if i != pred_class:
        margin_info.append((i, u_bounds[i]))

# Certification check
certified = all(l_bounds[pred_class] > u_bounds[i] for i in range(len(l_bounds)) if i != pred_class)

# Margin info
max_other = max(margin_info, key=lambda x: x[1])
margin = l_bounds[pred_class] - max_other[1]

print("\nCertification Result:")
if certified:
    print(f"Image is CERTIFIED ROBUST at ε = {epsilon}")
else:
    print(f"NOT CERTIFIED — prediction might change under perturbation.")

print(f"Closest competing class: {max_other[0]} with upper bound {max_other[1]:.4f}")
print(f"Required: Lower bound of class {pred_class} > {max_other[1]:.4f}")
print(f"Margin: {margin:.4f} {'(Safe)' if margin > 0 else 'Unsafe'}\n")

# Optional: suggest retrying with lower ε if needed
if not certified and epsilon > 0.0001:
    print(f" Tip: Try a smaller ε (e.g., {epsilon/2}) or a different test image (e.g., index = {index+1})")
