import json
import matplotlib.pyplot as plt
import numpy as np
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

from improved_layers import improved_conv_layer, improved_pool_layer
import time

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
images, labels = generate_dataset(50)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.3, random_state=246
)

fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
for i in range(4):
    ax[i // 2, i % 2].imshow(
        train_images[i].reshape(2, 4),  # Change back to 2 by 4
        aspect="equal",
    )
plt.subplots_adjust(wspace=0.1, hspace=0.025)

feature_map = ZFeatureMap(8)
feature_map.decompose().draw("mpl", style="clifford")

feature_map = ZFeatureMap(8)

def create_original_ansatz():
    """Create ansatz using original conv and pool layers."""
    ansatz = QuantumCircuit(8, name="Original Ansatz")
    
    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    
    # First Pooling Layer
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)
    
    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    
    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    
    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    
    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
    
    return ansatz

def create_improved_ansatz(layer_type="improved"):
    """Create ansatz using improved conv and pool layers."""
    ansatz = QuantumCircuit(8, name=f"Improved Ansatz ({layer_type})")
    
    # First Convolutional Layer
    ansatz.compose(improved_conv_layer(8, "c1", layer_type), list(range(8)), inplace=True)
    
    # First Pooling Layer
    ansatz.compose(improved_pool_layer([0,1,2,3], [4,5,6,7], "p1", "improved"), list(range(8)), inplace=True)
    
    # Second Convolutional Layer
    ansatz.compose(improved_conv_layer(4, "c2", layer_type), list(range(4, 8)), inplace=True)
    
    # Second Pooling Layer
    ansatz.compose(improved_pool_layer([0, 1], [2, 3], "p2", "improved"), list(range(4, 8)), inplace=True)
    
    # Third Convolutional Layer
    ansatz.compose(improved_conv_layer(2, "c3", layer_type), list(range(6, 8)), inplace=True)
    
    # Third Pooling Layer
    ansatz.compose(improved_pool_layer([0], [1], "p3", "improved"), list(range(6, 8)), inplace=True)
    
    return ansatz

def create_qnn_classifier(ansatz, name=""):
    """Create QNN classifier from ansatz."""
    # Combining the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    
    # Create QNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    
    # Create classifier
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=50, disp=False),  # Reduced iterations for faster comparison
    )
    
    return classifier, circuit, len(ansatz.parameters)

def compare_architectures():
    """Compare original vs improved layer architectures."""
    print("="*70)
    print("QCNN ARCHITECTURE COMPARISON")
    print("="*70)
    
    # Prepare data
    x_train = np.asarray(train_images)
    y_train = np.asarray(train_labels)
    x_test = np.asarray(test_images)
    y_test = np.asarray(test_labels)
    
    results = {}
    
    # Test different architectures
    architectures = [
        ("Original", create_original_ansatz()),
        ("Improved", create_improved_ansatz("improved")),
        ("Hardware Efficient", create_improved_ansatz("hardware_efficient")),
    ]
    
    for arch_name, ansatz in architectures:
        print(f"\n{'-'*50}")
        print(f"Testing: {arch_name}")
        print(f"{'-'*50}")
        
        try:
            # Create classifier
            classifier, circuit, num_params = create_qnn_classifier(ansatz, arch_name)
            
            print(f"Parameters: {num_params}")
            print(f"Circuit depth: {circuit.depth()}")
            print("Training...")
            
            # Train
            start_time = time.time()
            classifier.fit(x_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            train_acc = classifier.score(x_train, y_train)
            test_acc = classifier.score(x_test, y_test)
            
            # Store results
            results[arch_name] = {
                'parameters': num_params,
                'depth': circuit.depth(),
                'training_time': training_time,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'success': True
            }
            
            print(f"Training time: {training_time:.2f}s")
            print(f"Train accuracy: {100*train_acc:.2f}%")
            print(f"Test accuracy: {100*test_acc:.2f}%")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results[arch_name] = {
                'parameters': num_params if 'num_params' in locals() else 'N/A',
                'depth': 'N/A',
                'training_time': float('inf'),
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'success': False
            }
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Architecture':<18} {'Params':<8} {'Depth':<7} {'Time(s)':<8} {'Train%':<8} {'Test%':<8}")
    print(f"{'-'*70}")
    
    for arch_name, result in results.items():
        if result['success']:
            print(f"{arch_name:<18} {result['parameters']:<8} {result['depth']:<7} "
                  f"{result['training_time']:<8.1f} {100*result['train_accuracy']:<8.1f} "
                  f"{100*result['test_accuracy']:<8.1f}")
        else:
            print(f"{arch_name:<18} {'FAILED':<35}")
    
    print(f"{'='*70}")
    
    # Find best performing architecture
    successful_results = {k: v for k, v in results.items() if v['success']}
    if successful_results:
        best_arch = max(successful_results, key=lambda x: successful_results[x]['test_accuracy'])
        best_result = successful_results[best_arch]
        
        print(f"\n🏆 BEST ARCHITECTURE: {best_arch}")
        print(f"   Test Accuracy: {100*best_result['test_accuracy']:.2f}%")
        print(f"   Parameters: {best_result['parameters']}")
        print(f"   Training Time: {best_result['training_time']:.2f}s")
        
        # Calculate improvements over original (if original succeeded)
        if "Original" in successful_results and best_arch != "Original":
            orig_acc = successful_results["Original"]['test_accuracy']
            improvement = (best_result['test_accuracy'] - orig_acc) * 100
            print(f"   Improvement over Original: {improvement:+.2f} percentage points")
    
    return results

# Run the comparison
comparison_results = compare_architectures()