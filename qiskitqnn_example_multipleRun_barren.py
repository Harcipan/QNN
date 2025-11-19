"""
Qiskit example QNN running 100 times with 8 qubits 200iter COBYLA optimizer
WITH BARREN PLATEAU DIAGNOSTICS
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
from scipy.optimize import minimize_scalar

algorithm_globals.random_seed = 12345
estimator = Estimator()
# We now define a two qubit unitary as defined in [3]
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


# Let's draw this circuit and see what it looks like
params = ParameterVector("θ", length=3)
circuit = conv_circuit(params)
circuit.draw("mpl", style="clifford")
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

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

# ========== BARREN PLATEAU DIAGNOSTIC FUNCTIONS ==========

def compute_finite_difference_gradient(qnn, params, X, y, h=1e-4):
    """Compute gradient using finite differences"""
    n_params = len(params)
    gradient = np.zeros(n_params)
    
    # Define loss function
    def loss_func(theta):
        # Get predictions from QNN
        predictions = []
        for x_sample in X:
            pred = qnn.forward(x_sample, theta)
            predictions.append(pred[0] if hasattr(pred, '__len__') else pred)
        predictions = np.array(predictions)
        
        # Binary cross-entropy loss
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        y_prob = (predictions + 1) / 2  # Convert from [-1,1] to [0,1]
        y_binary = (y + 1) / 2  # Convert labels from [-1,1] to [0,1]
        loss = -np.mean(y_binary * np.log(y_prob) + (1 - y_binary) * np.log(1 - y_prob))
        return loss
    
    # Compute base loss
    base_loss = loss_func(params)
    
    # Compute gradient using finite differences
    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += h
        loss_plus = loss_func(params_plus)
        gradient[i] = (loss_plus - base_loss) / h
    
    return gradient, base_loss

def gradient_diagnostics(qnn, params, X, y, run_id, checkpoint_name):
    """Compute gradient statistics for barren plateau detection"""
    gradient, loss = compute_finite_difference_gradient(qnn, params, X, y)
    
    # Gradient statistics
    grad_norm = np.linalg.norm(gradient)
    grad_mean = np.mean(np.abs(gradient))
    grad_median = np.median(np.abs(gradient))
    grad_var = np.var(gradient)
    grad_max = np.max(np.abs(gradient))
    
    stats = {
        'run_id': run_id,
        'checkpoint': checkpoint_name,
        'loss': loss,
        'grad_norm': grad_norm,
        'grad_mean': grad_mean,
        'grad_median': grad_median, 
        'grad_var': grad_var,
        'grad_max': grad_max,
        'n_params': len(params)
    }
    
    return stats, gradient

def random_direction_line_scan(qnn, params, X, y, n_scans=3, n_points=21, scan_range=0.5):
    """Perform random direction line scans to check loss landscape"""
    
    def loss_func(theta):
        predictions = []
        for x_sample in X:
            pred = qnn.forward(x_sample, theta)
            predictions.append(pred[0] if hasattr(pred, '__len__') else pred)
        predictions = np.array(predictions)
        
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        y_prob = (predictions + 1) / 2
        y_binary = (y + 1) / 2
        loss = -np.mean(y_binary * np.log(y_prob) + (1 - y_binary) * np.log(1 - y_prob))
        return loss
    
    scan_results = []
    
    for scan_idx in range(n_scans):
        # Generate random direction
        direction = np.random.randn(len(params))
        direction = direction / np.linalg.norm(direction)
        
        # Create line scan points
        alphas = np.linspace(-scan_range, scan_range, n_points)
        losses = []
        
        for alpha in alphas:
            theta_scan = params + alpha * direction
            loss = loss_func(theta_scan)
            losses.append(loss)
        
        scan_results.append({
            'scan_id': scan_idx,
            'alphas': alphas,
            'losses': losses,
            'direction': direction
        })
    
    return scan_results

def multi_seed_gradient_analysis(qnn_factory, X, y, n_seeds=5):
    """Run gradient analysis across multiple random seeds"""
    seed_results = []
    
    for seed in range(n_seeds):
        np.random.seed(seed + 42)  # Different seed for initialization
        
        # Initialize random parameters
        n_params = qnn_factory().num_weights
        params = np.random.uniform(-np.pi, np.pi, n_params)
        
        qnn = qnn_factory()
        stats, gradient = gradient_diagnostics(qnn, params, X, y, f"seed_{seed}", "init")
        stats['seed'] = seed
        seed_results.append(stats)
    
    return seed_results

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

# Initialize barren plateau diagnostics storage
all_gradient_stats = []
all_line_scans = []
multi_seed_results = []

# Run simulation 100 times
num_runs = 5  # Reduced for diagnostic testing
print(f"Starting {num_runs} simulation runs with barren plateau diagnostics...")
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
        feature_map = ZFeatureMap(8)
        
        ansatz = QuantumCircuit(8, name="Ansatz")
        
        # First Convolutional Layer
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        
        # First Pooling Layer
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        
        # Second Convolutional Layer
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        
        # Second Pooling Layer
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        
        # Third Convolutional Layer
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        
        # Third Pooling Layer
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
        
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
        
        # QNN factory function for multi-seed analysis
        def qnn_factory():
            return qnn

    # ========== BARREN PLATEAU DIAGNOSTICS ==========
    
    # Multi-seed gradient analysis (only on first run)
    if run == 0:
        print("  Running multi-seed gradient analysis...")
        seed_results = multi_seed_gradient_analysis(qnn_factory, 
                                                   np.asarray(train_images), 
                                                   np.asarray(train_labels))
        multi_seed_results.extend(seed_results)
    
    # Initial gradient diagnostics
    print("  Computing initial gradient diagnostics...")
    initial_params = np.random.uniform(-np.pi, np.pi, qnn.num_weights)
    stats_init, grad_init = gradient_diagnostics(qnn, initial_params, 
                                                 np.asarray(train_images), 
                                                 np.asarray(train_labels), 
                                                 run, "initial")
    all_gradient_stats.append(stats_init)
    
    # Initial random direction line scans
    print("  Performing initial line scans...")
    line_scans_init = random_direction_line_scan(qnn, initial_params,
                                                np.asarray(train_images),
                                                np.asarray(train_labels))
    for scan in line_scans_init:
        scan['run_id'] = run
        scan['checkpoint'] = 'initial'
    all_line_scans.extend(line_scans_init)

    def callback_graph_with_diagnostics(weights, obj_func_eval):
        # Store objective function values
        objective_func_vals.append(obj_func_eval)
        
        # Perform diagnostics every 50 iterations
        iteration = len(objective_func_vals)
        if iteration % 50 == 0:
            print(f"    Iteration {iteration}: Running diagnostics...")
            
            # Gradient diagnostics
            stats, gradient = gradient_diagnostics(qnn, weights,
                                                  np.asarray(train_images),
                                                  np.asarray(train_labels),
                                                  run, f"iter_{iteration}")
            all_gradient_stats.append(stats)
            
            # Line scans (fewer for performance)
            if iteration % 100 == 0:
                line_scans = random_direction_line_scan(qnn, weights,
                                                      np.asarray(train_images),
                                                      np.asarray(train_labels),
                                                      n_scans=2)  # Reduced for performance
                for scan in line_scans:
                    scan['run_id'] = run
                    scan['checkpoint'] = f"iter_{iteration}"
                all_line_scans.extend(line_scans)

    # Create classifier for this run
    objective_func_vals = []
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=200),  # Set max iterations here
        callback=callback_graph_with_diagnostics
    )

    # Prepare training data
    x = np.asarray(train_images)
    y = np.asarray(train_labels)

    # Train the classifier
    print("  Training classifier...")
    classifier.fit(x, y)
    
    # Final gradient diagnostics
    print("  Computing final gradient diagnostics...")
    final_params = classifier.weights
    stats_final, grad_final = gradient_diagnostics(qnn, final_params, x, y, run, "final")
    all_gradient_stats.append(stats_final)
    
    # Final random direction line scans
    print("  Performing final line scans...")
    line_scans_final = random_direction_line_scan(qnn, final_params, x, y)
    for scan in line_scans_final:
        scan['run_id'] = run
        scan['checkpoint'] = 'final'
    all_line_scans.extend(line_scans_final)

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
# Perform multi-seed gradient analysis if possible
# multi_seed_gradient_analysis expects (qnn_factory, X, y, n_seeds=...)
if 'qnn_factory' in globals() and len(train_accuracies) > 0:
    try:
        print("\nPerforming multi-seed gradient analysis using qnn_factory and last training set...")
        multi_seed_data = multi_seed_gradient_analysis(qnn_factory,
                                                     np.asarray(train_images),
                                                     np.asarray(train_labels))
        # extend with returned per-seed stats
        multi_seed_results.extend(multi_seed_data)
    except Exception as e:
        print(f"Failed to run multi-seed gradient analysis: {e}")
        # keep going without crashing
else:
    print("\nSkipping multi-seed gradient analysis: qnn_factory or training data not available.")

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

# Save all results to JSON for comprehensive analysis
comprehensive_results = {
    'runs': {
        'training_accuracies': train_accuracies.tolist(),
        'test_accuracies': test_accuracies.tolist(),
        'objective_values': all_objective_vals
    },
    'diagnostics': {
        'gradient_stats': all_gradient_stats,
        'line_scans': all_line_scans,
        'multi_seed_analysis': multi_seed_results
    },
    'metadata': {
        'total_runs': num_runs,
        'total_time_minutes': total_time/60,
        'qubits': 8,
        'optimizer': 'COBYLA'
    }
}

# Save comprehensive diagnostics
timestamp = int(time.time())
diag_filename = f'qcnn_barren_diagnostics_{num_runs}runs_{timestamp}.json'
with open(diag_filename, 'w') as f:
    json.dump(comprehensive_results, f, indent=2)

# Save results to CSV file (for compatibility)
results_df = pd.DataFrame({
    'Run': range(1, num_runs + 1),
    'Train_Accuracy': train_accuracies,
    'Test_Accuracy': test_accuracies
})

csv_filename = f'qcnn_barren_results_{num_runs}runs_{timestamp}.csv'
results_df.to_csv(csv_filename, index=False)

print(f"\nResults saved:")
print(f"- CSV file: {csv_filename}")
print(f"- Comprehensive diagnostics: {diag_filename}")

# Print barren plateau diagnostic summary
if all_gradient_stats:
    print(f"\n{'='*50}")
    print(f"BARREN PLATEAU DIAGNOSTIC SUMMARY")
    print(f"{'='*50}")
    print(f"- Total gradient evaluations: {len(all_gradient_stats)}")
    print(f"- Total line scans: {len(all_line_scans)}")
    if multi_seed_results:
        print(f"- Multi-seed analyses: {len(multi_seed_results)}")
        
    # Summary statistics
    grad_norms = [stats['gradient_norm'] for stats in all_gradient_stats]
    grad_max_components = [stats['max_component'] for stats in all_gradient_stats]
    
    print(f"\nGradient Norm Analysis:")
    print(f"- Range: {min(grad_norms):.6f} to {max(grad_norms):.6f}")
    print(f"- Average: {np.mean(grad_norms):.6f} ± {np.std(grad_norms):.6f}")
    
    print(f"\nGradient Component Analysis:")
    print(f"- Max component range: {min(grad_max_components):.6f} to {max(grad_max_components):.6f}")
    print(f"- Max component average: {np.mean(grad_max_components):.6f}")
    
    # Barren plateau indicators
    barren_threshold = 1e-6
    barren_count = sum(1 for norm in grad_norms if norm < barren_threshold)
    print(f"\nBarren Plateau Indicators:")
    print(f"- Gradient norms < {barren_threshold}: {barren_count}/{len(grad_norms)} ({100*barren_count/len(grad_norms):.1f}%)")
    
    if barren_count > len(grad_norms) * 0.5:
        print("⚠️  WARNING: High likelihood of barren plateau detected!")
    elif barren_count > len(grad_norms) * 0.2:
        print("⚠️  CAUTION: Moderate barren plateau indicators detected")
    else:
        print("✅ Low barren plateau risk detected")

# Create and save enhanced plot with diagnostics
plt.figure(figsize=(16, 10))

# Accuracy plots
plt.subplot(2, 3, 1)
plt.hist(100 * train_accuracies, bins=20, alpha=0.7, label='Train', color='blue')
plt.hist(100 * test_accuracies, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Accuracy (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracies')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(range(1, num_runs + 1), 100 * train_accuracies, 'b-', alpha=0.6, label='Train')
plt.plot(range(1, num_runs + 1), 100 * test_accuracies, 'r-', alpha=0.6, label='Test')
plt.xlabel('Run Number')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Run Number')
plt.legend()

# Diagnostic plots
if all_gradient_stats:
    # Gradient norm over time
    plt.subplot(2, 3, 3)
    grad_norms = [stats['gradient_norm'] for stats in all_gradient_stats]
    checkpoints = [stats['checkpoint'] for stats in all_gradient_stats]
    runs = [stats['run_id'] for stats in all_gradient_stats]
    
    plt.semilogy(grad_norms, 'o-', alpha=0.7)
    plt.xlabel('Gradient Evaluation')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Norm Evolution')
    plt.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Barren threshold')
    plt.legend()
    
    # Gradient component distribution
    plt.subplot(2, 3, 4)
    max_components = [stats['max_component'] for stats in all_gradient_stats]
    plt.hist(np.log10(np.abs(max_components)), bins=15, alpha=0.7)
    plt.xlabel('log10(Max Gradient Component)')
    plt.ylabel('Frequency')
    plt.title('Gradient Component Distribution')

# Line scan results if available
if all_line_scans:
    plt.subplot(2, 3, 5)
    for i, scan in enumerate(all_line_scans[:10]):  # Show first 10 scans
        plt.plot(scan['alphas'], scan['losses'], alpha=0.5)
    plt.xlabel('Alpha (step size)')
    plt.ylabel('Loss')
    plt.title('Random Direction Line Scans')

# Accuracy correlation
plt.subplot(2, 3, 6)
plt.scatter(100 * train_accuracies, 100 * test_accuracies, alpha=0.7)
plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)
plt.xlabel('Training Accuracy (%)')
plt.ylabel('Test Accuracy (%)')
plt.title('Test vs Training Accuracy')

plt.tight_layout()
plot_filename = f'qcnn_barren_analysis_{num_runs}runs_{timestamp}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Enhanced plot saved to: {plot_filename}")
plt.show()

print(f"\n{'='*50}")
print(f"BARREN PLATEAU ANALYSIS COMPLETE!")
print(f"{'='*50}")