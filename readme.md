# 1D Swift–Hohenberg PINN Solver

This project implements numerical solutions to the 1D Swift-Hohenberg equation, a fundamental PDE for pattern formation studies. Two distinct approaches are provided:

Spectral Method: A traditional solver using Fourier transforms and Crank-Nicholson time-stepping

Physics-Informed Neural Network (PINN): A machine learning approach that trains a neural network to satisfy the PDE physics.


Key features:

Models pattern formation from a perturbed cosine initial condition

Parameters: Lx=64π domain, r=0.2 (bifurcation), b₂=1.2 (nonlinearity)

Includes visualization of temperature patterns over time

PINN implementation features automatic checkpointing and loss tracking

---

## 📁 Project Structure

project_folder/
│
├── pinn_model.py # Defines the neural network architecture
├── main_experiment.py # Trains the PINN and stores model checkpoints and losses
├── plot_log_loss.py # Visualizes log-scale loss components over epochs
├── plot_normalised_loss.py # Visualizes normalized loss components over epochs
├── pattern.py # Plots temperature patterns for different epochs
│
├── checkpoints/ # Saved models and training logs
├── results/ # Plots for loss components
├── pattern_evolution/ # PNG snapshots and animations of pattern evolution
│
└── README.md # This file



---

## 🚀 How to Run

Make sure you have Python 3.8+ and required packages installed. You can create a virtual environment and install dependencies.


| Folder               | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `checkpoints/`       | Contains model weights and loss logs for each epoch. |
| `results/`           | Loss component plots over training epochs.           |
| `pattern_evolution/` | PNG images and animations of solution patterns.      |

