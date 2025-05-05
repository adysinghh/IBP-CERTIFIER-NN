# IBP-CERTIFIER-NN

A lightweight and practical project to **train a neural network on MNIST** and **formally certify its robustness** using **Interval Bound Propagation (IBP)**. The goal? To answer a simple but important question:

> Can we guarantee that the model's prediction will not change even if the input image is slightly perturbed?

Instead of relying on adversarial attacks, we use formal analysis to compute bounds — no tricks, no hacks, just math.

---

## What This Project Does

- Trains a fully connected neural network on MNIST (99.9% accuracy achieved)
- Applies Interval Bound Propagation (IBP) to mathematically track how input noise affects output logits
- Verifies whether the predicted class remains stable within a given perturbation radius (ε)
- Outputs bounds, margins, and certification verdict for each image
- Optionally visualizes certified vs. uncertified examples

---

## How to Use

### 1. Clone the repository

```bash
git clone https://github.com/adysinghh/IBP-CERTIFIER-NN.git
cd IBP-CERTIFIER-NN
````

---

### 2. Set up environment

```bash
pip install -r requirements.txt
```

---

### 3. Train the model

```bash
python train_model.py
```

Example training output:

```
Epoch 50 | Loss: 0.0037 | Accuracy: 99.91%
Model training complete and saved to 'model.pth'
```

---

### 4. Run the certifier

```bash
python certifier.py
```

Example certification output:

```
Clean Prediction: 2 (Confidence: 49.9437) | True Label: 2 | Epsilon: 0.001
Prediction CORRECT

Output Logit Bounds:
Class 2: Lower = 40.5115, Upper = 58.6902
Class 1: Upper = -5.1823
...

Image is CERTIFIED ROBUST at ε = 0.001
Margin: 45.6938 (Safe)
```

---

## Why Certification Matters

In real-world scenarios — whether it's autonomous vehicles or medical AI — we need guarantees, not guesses.
This project shows how we can **mathematically certify** when a model is making a reliable prediction.

---

## Project Structure

```
IBP-CERTIFIER-NN/
├── train_model.py              # Trains the neural net on MNIST
├── certifier.py                # Runs IBP and certifies a prediction
├── interval_propagation.py     # Core logic to compute interval bounds
├── visualize.py                # (Optional) shows image + cert status
├── requirements.txt
└── README.md
```
---

## Acknowledgements

* Inspired by recent research on certified AI and robustness verification
* Special thanks to Dr. Xiyue Zhang and Marta Kwiatkowska’s work on formal verification of deep networks

---
