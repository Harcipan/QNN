"""
Qiskit example QNN running 100 times with 8 qubits 200iter COBYLA optimizer
Multi-layer architecture: 8->8->4->4->2->2 qubits with duplicated conv/pool layers
Using IMPROVED convolution circuits with 6 parameters for enhanced expressivity
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
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from improved_layers import improved_conv_circuit

algorithm_globals.random_seed = 12345
estimator = Estimator()
# Using improved convolution circuit with enhanced expressivity (6 parameters)
def conv_circuit(params):
    """
    Enhanced convolution circuit with more expressivity.
    Uses 6 parameters instead of 3 for richer parameterization.
    """
    return improved_conv_circuit(params)


# Let's draw this circuit and see what it looks like
params = ParameterVector("θ", length=6)
circuit = conv_circuit(params)
circuit.draw("mpl", style="clifford")
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Improved Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 6)  # 6 parameters per gate
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 6)]), [q1, q2])
        qc.barrier()
        param_index += 6
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 6)]), [q1, q2])
        qc.barrier()
        param_index += 6

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


circuit = conv_layer(4, "θ")
circuit.decompose().draw("mpl", style="clifford")

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


params = ParameterVector("θ", length=3)
circuit = pool_circuit(params)
circuit.draw("mpl", style="clifford")

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


sources = [0, 1]
sinks = [2, 3]
circuit = pool_layer(sources, sinks, "θ")
circuit.decompose().draw("mpl", style="clifford")

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

# Run simulation 100 times with multi-layer architecture
num_runs = 100
print(f"Starting {num_runs} simulation runs with MULTI-LAYER architecture...")
print("Architecture: 8->8->4->4->2->2 qubits (duplicated conv/pool layers at each level)")
print("Using IMPROVED convolution circuits with 6 parameters per gate for enhanced expressivity")
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
        print("  Building multi-layer quantum circuit with IMPROVED convolution...")
        feature_map = ZFeatureMap(8)
        
        ansatz = QuantumCircuit(8, name="Multi-Layer Ansatz with Improved Conv")
        
        # First Convolutional Layer (8 qubits) - IMPROVED
        print("    Adding improved conv layer 1 (8 qubits)...")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        
        # First Pooling Layer (8 qubits)
        print("    Adding pool layer 1 (8 qubits)...")
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        
        # Second Convolutional Layer (8 qubits) - DUPLICATE IMPROVED
        print("    Adding improved conv layer 2 (8 qubits) - duplicate...")
        ansatz.compose(conv_layer(8, "c2"), list(range(8)), inplace=True)
        
        # Second Pooling Layer (8 qubits) - DUPLICATE
        print("    Adding pool layer 2 (8 qubits) - duplicate...")
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"), list(range(8)), inplace=True)
        
        # Third Convolutional Layer (4 qubits) - IMPROVED
        print("    Adding improved conv layer 3 (4 qubits)...")
        ansatz.compose(conv_layer(4, "c3"), list(range(4, 8)), inplace=True)
        
        # Third Pooling Layer (4 qubits)
        print("    Adding pool layer 3 (4 qubits)...")
        ansatz.compose(pool_layer([0, 1], [2, 3], "p3"), list(range(4, 8)), inplace=True)
        
        # Fourth Convolutional Layer (4 qubits) - DUPLICATE IMPROVED
        print("    Adding improved conv layer 4 (4 qubits) - duplicate...")
        ansatz.compose(conv_layer(4, "c4"), list(range(4, 8)), inplace=True)
        
        # Fourth Pooling Layer (4 qubits) - DUPLICATE
        print("    Adding pool layer 4 (4 qubits) - duplicate...")
        ansatz.compose(pool_layer([0, 1], [2, 3], "p4"), list(range(4, 8)), inplace=True)
        
        # Fifth Convolutional Layer (2 qubits) - IMPROVED
        print("    Adding improved conv layer 5 (2 qubits)...")
        ansatz.compose(conv_layer(2, "c5"), list(range(6, 8)), inplace=True)
        
        # Fifth Pooling Layer (2 qubits)
        print("    Adding pool layer 5 (2 qubits)...")
        ansatz.compose(pool_layer([0], [1], "p5"), list(range(6, 8)), inplace=True)
        
        # Sixth Convolutional Layer (2 qubits) - DUPLICATE IMPROVED
        print("    Adding improved conv layer 6 (2 qubits) - duplicate...")
        ansatz.compose(conv_layer(2, "c6"), list(range(6, 8)), inplace=True)
        
        # Sixth Pooling Layer (2 qubits) - DUPLICATE
        print("    Adding pool layer 6 (2 qubits) - duplicate...")
        ansatz.compose(pool_layer([0], [1], "p6"), list(range(6, 8)), inplace=True)
        
        print(f"    Total circuit parameters: {len(circuit.parameters)}")
        print(f"    Feature map parameters: {len(feature_map.parameters)}")
        print(f"    Ansatz parameters: {len(ansatz.parameters)}")
        print(f"    Multi-layer architecture: 8->8->4->4->2->2 qubits with IMPROVED conv circuits")
        print(f"    Each conv layer uses 6 parameters for enhanced expressivity")
        
        # Combining the feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        
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

    # Create classifier for this run
    objective_func_vals = []
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=200),  # Set max iterations here
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

filename = f'qcnn_improved_multilayer_8qubit_{num_runs}runs_results_{int(time.time())}.csv'
results_df.to_csv(filename, index=False)
print(f"\nResults saved to: {filename}")

# Create and save summary plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(100 * train_accuracies, bins=20, alpha=0.7, label='Train', color='blue')
plt.hist(100 * test_accuracies, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracies\n(Improved Multi-Layer: 8->8->4->4->2->2 qubits)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_runs + 1), 100 * train_accuracies, 'b-', alpha=0.6, label='Train')
plt.plot(range(1, num_runs + 1), 100 * test_accuracies, 'r-', alpha=0.6, label='Test')
plt.xlabel('Run Number')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Run Number\n(Improved Conv Duplicated Architecture)')
plt.legend()

plt.tight_layout()
plot_filename = f'qcnn_improved_multilayer_8qubit_{num_runs}runs_plot_{int(time.time())}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_filename}")
print(f"\n🏗️  IMPROVED DUPLICATED MULTI-LAYER ARCHITECTURE SUMMARY:")
print(f"   8 qubits -> ImprovedConv1 -> Pool1 -> ImprovedConv2 -> Pool2")
print(f"   4 qubits -> ImprovedConv3 -> Pool3 -> ImprovedConv4 -> Pool4")  
print(f"   2 qubits -> ImprovedConv5 -> Pool5 -> ImprovedConv6 -> Pool6")
print(f"   Total layers: 6 Improved Conv + 6 Pool = 12 layers")
print(f"   Each conv layer uses 6 parameters for enhanced expressivity")
print(f"   Improved circuits have: RY+RY+CX+RY+RY+RZ+RZ+CX structure")
plt.show()