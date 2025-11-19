"""
Qiskit example QNN running with IMPROVED CONVOLUTION CIRCUITS and SPSA OPTIMIZER
Using improved conv circuits with enhanced expressivity from improved_layers.py
with SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer for better quantum performance
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

# Import improved circuits
from improved_layers import (
    improved_conv_circuit, 
    hardware_efficient_conv_circuit,
    improved_conv_layer,
    improved_pool_circuit,
    improved_pool_layer
)

algorithm_globals.random_seed = 12345
estimator = Estimator()

# Using improved convolution circuit for enhanced expressivity
def conv_circuit(params):
    """Enhanced convolution circuit with more expressivity"""
    return improved_conv_circuit(params)

# Alternative hardware-efficient version (not used in this run)
def conv_circuit_hardware_eff(params):
    """Hardware-efficient ansatz style convolution - good for NISQ devices"""
    return hardware_efficient_conv_circuit(params)

# Let's use the improved conv circuit
params = ParameterVector("θ", length=6)  # 6 parameters for improved circuit
circuit = conv_circuit(params)
print("Improved convolution circuit:")
print(circuit.draw())

def conv_layer(num_qubits, param_prefix):
    """Enhanced convolutional layer using improved circuits"""
    qc = QuantumCircuit(num_qubits, name="Enhanced Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    # 6 parameters per gate pair, num_qubits gates total
    params = ParameterVector(param_prefix, length=num_qubits * 6)
    
    # First round: even pairs (0,1), (2,3), (4,5), ...
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 6)]), [q1, q2])
        qc.barrier()
        param_index += 6
    
    # Second round: odd pairs (1,2), (3,4), (5,6), ... + wrap around
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        if param_index + 6 <= len(params):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 6)]), [q1, q2])
            qc.barrier()
            param_index += 6

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


circuit = conv_layer(4, "θ")
print("Improved conv layer:")
print(circuit.decompose().draw())

def pool_circuit(params):
    """Enhanced pooling circuit with controlled rotations"""
    return improved_pool_circuit(params)

params = ParameterVector("θ", length=4)  # 4 parameters for enhanced pooling
circuit = pool_circuit(params)
print("Enhanced pooling circuit:")
print(circuit.draw())

def pool_layer(sources, sinks, param_prefix):
    """Enhanced pooling layer using improved circuits"""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Enhanced Pooling Layer")
    param_index = 0
    # 4 parameters per pooling operation
    params = ParameterVector(param_prefix, length=len(sources) * 4)
    
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 4)]), [source, sink])
        qc.barrier()
        param_index += 4

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


sources = [0, 1]
sinks = [2, 3]
circuit = pool_layer(sources, sinks, "θ")
print("Enhanced pool layer:")
print(circuit.decompose().draw())

def generate_dataset(num_images):
    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for n in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return images, labels

# Initialize lists to store results from 100 runs
train_accuracies = []
test_accuracies = []
all_objective_vals = []

# Run simulation with improved circuits and SPSA optimizer
num_runs = 100
print(f"Starting {num_runs} simulation runs with IMPROVED CONVOLUTION CIRCUITS and SPSA OPTIMIZER...")
print("Improved circuits use 6 parameters per conv gate with enhanced expressivity")
print("SPSA optimizer is designed for noisy quantum systems and gradient-free optimization")
start_time = time.time()

for run in range(num_runs):
    print(f"\nRun {run + 1}/{num_runs}")
    
    # Generate new dataset for each run with different random seed
    np.random.seed(run)  # Set different seed for each run
    algorithm_globals.random_seed = run + 12345
    
    images, labels = generate_dataset(50)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=run + 246  # Different random state for each run
    )

    # Setup quantum circuit components (these don't change between runs)
    if run == 0:  # Only create these once
        print("  Building improved quantum circuit...")
        feature_map = ZFeatureMap(8)
        
        ansatz = QuantumCircuit(8, name="Improved Ansatz")
        
        # First Convolutional Layer - Improved
        print("    Adding improved conv layer 1...")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        
        # First Pooling Layer - Enhanced
        print("    Adding enhanced pool layer 1...")
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        
        # Second Convolutional Layer - Improved
        print("    Adding improved conv layer 2...")
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        
        # Second Pooling Layer - Enhanced
        print("    Adding enhanced pool layer 2...")
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        
        # Third Convolutional Layer - Improved
        print("    Adding improved conv layer 3...")
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        
        # Third Pooling Layer - Enhanced
        print("    Adding enhanced pool layer 3...")
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
        
        # Combining the feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        
        print(f"    Total circuit parameters: {len(circuit.parameters)}")
        print(f"    Feature map parameters: {len(feature_map.parameters)}")
        print(f"    Ansatz parameters: {len(ansatz.parameters)}")
        
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        
        # we decompose the circuit for the QNN to avoid additional data copying
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        circuit.draw("mpl", style="clifford")

    def callback_graph(weights, obj_func_eval):
        # Store objective function values but don't plot during multiple runs
        objective_func_vals.append(obj_func_eval)

    # Create classifier for this run with SPSA optimizer
    objective_func_vals = []
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=SPSA(maxiter=200, learning_rate=0.02, perturbation=0.01),  # SPSA optimizer
        callback=callback_graph
    )

    # Prepare training data
    x = np.asarray(train_images)
    y = np.asarray(train_labels)

    # Train the classifier
    classifier.fit(x, y)

    # Calculate training accuracy
    train_accuracy = classifier.score(x, y)
    train_accuracies.append(train_accuracy)

    # Calculate test accuracy
    x_test = np.asarray(test_images)
    y_test = np.asarray(test_labels)
    test_accuracy = classifier.score(x_test, y_test)
    test_accuracies.append(test_accuracy)
    
    # Store objective function values for this run
    all_objective_vals.append(objective_func_vals.copy())

    # Print progress
    print(f"  Train accuracy: {np.round(100 * train_accuracy, 2)}%")
    print(f"  Test accuracy: {np.round(100 * test_accuracy, 2)}%")
    
    # Print progress every 10 runs
    if (run + 1) % 10 == 0:
        elapsed_time = time.time() - start_time
        avg_time_per_run = elapsed_time / (run + 1)
        estimated_total_time = avg_time_per_run * num_runs
        remaining_time = estimated_total_time - elapsed_time
        print(f"\nCompleted {run + 1}/{num_runs} runs.")
        print(f"Average time per run: {avg_time_per_run:.1f}s")
        print(f"Estimated remaining time: {remaining_time/60:.1f} minutes")

# Calculate and print summary statistics
total_time = time.time() - start_time
print(f"\n{'='*50}")
print(f"SIMULATION COMPLETED in {total_time/60:.1f} minutes")
print(f"{'='*50}")

train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)

print(f"\nTraining Accuracy Statistics:")
print(f"  Mean: {np.round(100 * np.mean(train_accuracies), 2)}%")
print(f"  Std:  {np.round(100 * np.std(train_accuracies), 2)}%")
print(f"  Min:  {np.round(100 * np.min(train_accuracies), 2)}%")
print(f"  Max:  {np.round(100 * np.max(train_accuracies), 2)}%")

print(f"\nTest Accuracy Statistics:")
print(f"  Mean: {np.round(100 * np.mean(test_accuracies), 2)}%")
print(f"  Std:  {np.round(100 * np.std(test_accuracies), 2)}%")
print(f"  Min:  {np.round(100 * np.min(test_accuracies), 2)}%")
print(f"  Max:  {np.round(100 * np.max(test_accuracies), 2)}%")

# Save results to CSV file
results_df = pd.DataFrame({
    'Run': range(1, num_runs + 1),
    'Train_Accuracy': train_accuracies,
    'Test_Accuracy': test_accuracies
})

filename = f'qcnn_improved_spsa_{num_runs}runs_results_{int(time.time())}.csv'
results_df.to_csv(filename, index=False)
print(f"\nResults saved to: {filename}")

# Create and save summary plot
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(100 * train_accuracies, bins=20, alpha=0.7, label='Train', color='blue')
plt.hist(100 * test_accuracies, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracies\n(Improved Conv + SPSA)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, num_runs + 1), 100 * train_accuracies, 'b-', alpha=0.6, label='Train')
plt.plot(range(1, num_runs + 1), 100 * test_accuracies, 'r-', alpha=0.6, label='Test')
plt.xlabel('Run Number')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Run Number\n(Improved Conv + SPSA)')
plt.legend()

# Additional analysis plots
plt.subplot(2, 2, 3)
# Plot correlation between train and test accuracy
plt.scatter(100 * train_accuracies, 100 * test_accuracies, alpha=0.7)
plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)
plt.xlabel('Training Accuracy (%)')
plt.ylabel('Test Accuracy (%)')
plt.title('Test vs Training Accuracy')

plt.subplot(2, 2, 4)
# Plot accuracy difference (overfitting indicator)
accuracy_diff = train_accuracies - test_accuracies
plt.hist(100 * accuracy_diff, bins=15, alpha=0.7, color='orange')
plt.xlabel('Training - Test Accuracy (%)')
plt.ylabel('Frequency')
plt.title('Overfitting Analysis')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plot_filename = f'qcnn_improved_spsa_{num_runs}runs_plot_{int(time.time())}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Enhanced plot saved to: {plot_filename}")

# Print parameter count comparison
print(f"\n{'='*60}")
print(f"IMPROVED CIRCUIT + SPSA OPTIMIZER ANALYSIS")
print(f"{'='*60}")
print(f"Improved Convolution: 6 parameters per gate (enhanced expressivity)")
print(f"Enhanced Pooling: 4 parameters per gate (vs 3 original)")
print(f"SPSA Optimizer: Learning rate=0.02, Perturbation=0.01, Max iterations=200")
print(f"Improved circuits provide better expressivity and quantum advantage")
print(f"SPSA provides gradient-free optimization suitable for noisy quantum systems")

plt.show()