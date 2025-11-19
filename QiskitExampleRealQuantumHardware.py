"""
Quantum Neural Network Image Classification on IBM Quantum Hardware
Classifies horizontal vs vertical line patterns using QCNN on real quantum computers
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from IPython.display import clear_output

# Core Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

# IBM Quantum Runtime imports  
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeFez

# Machine Learning imports
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split

from qiskit_ibm_runtime import QiskitRuntimeService
 
QiskitRuntimeService.save_account(
    token="", 
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/90ac20c18f6f4077b1d0e84d774560c2:436c3853-3350-418c-9ef5-8bac3f6cda22::",
    overwrite=True  # Add this to overwrite existing account
)

from qiskit_ibm_runtime import QiskitRuntimeService

# Test the connection
service = QiskitRuntimeService()
print("✅ IBM Quantum service activated successfully!")
print(f"Available backends: {len(service.backends())}")

# List some available backends
backends = service.backends(simulator=False, operational=True)
print("\n🖥️ Available quantum computers:")
for backend in backends[:5]:  # Show first 5
    print(f"  - {backend.name}: {backend.configuration().n_qubits} qubits")

# ============ HARDWARE CONFIGURATION ============
USE_HARDWARE = True  # Set to False for simulation testing
HARDWARE_BACKEND = "ibm_torino"  # or "ibm_brisbane", "ibm_kyoto", etc.
NUM_IMAGES = 20  # Reduced for hardware (was 50)
MAX_ITERATIONS = 30  # Reduced for hardware (was 200)
TEST_SIZE = 0.3

print("🚀 IBM Quantum Neural Network Image Classifier")
print("=" * 60)

# Set random seed for reproducibility
algorithm_globals.random_seed = 12345

# ============ HARDWARE CONNECTION SETUP ============
def setup_quantum_backend():
    """Setup IBM Quantum hardware connection with fallback to simulator"""
    try:
        if USE_HARDWARE:
            service = QiskitRuntimeService()
            
            # Get available backends
            backends = service.backends(simulator=False, operational=True)
            print("🖥️ Available quantum computers:")
            for backend in backends[:5]:
                status = backend.status()
                print(f"  - {backend.name}: {status.pending_jobs} jobs queued, "
                      f"{backend.configuration().n_qubits} qubits")
            
            # Try to get the specified backend, fallback to least busy
            try:
                backend = service.backend(HARDWARE_BACKEND)
                print(f"\n✅ Selected backend: {backend.name}")
            except:
                backend = service.least_busy(simulator=False, operational=True)
                print(f"\n✅ Selected least busy backend: {backend.name}")
            
            print(f"Number of qubits: {backend.configuration().n_qubits}")
            print(f"Queue length: {backend.status().pending_jobs}")
            
            # Configure estimator for job mode (compatible with Open Plan)
            estimator = Estimator(mode=backend)
            estimator.options.resilience_level = 1  # Basic error mitigation
            estimator.options.default_shots = 1024  # Balanced shots for QNN
            
            print(f"✅ Hardware estimator configured in job mode:")
            print(f"  - Execution mode: Job (Open Plan compatible)")
            print(f"  - Resilience level: {estimator.options.resilience_level}")
            print(f"  - Shots per execution: {estimator.options.default_shots}")
            
            return backend, estimator, True, None
            
        else:
            print("🔬 Using simulation mode...")
            backend = FakeFez()
            estimator = Estimator(backend)
            return backend, estimator, False, None
            
    except Exception as e:
        print(f"⚠️ Could not connect to IBM Quantum: {e}")
        print("🔬 Falling back to simulator...")
        backend = FakeFez()
        estimator = Estimator(backend)
        return backend, estimator, False, None

# Setup backend and estimator
backend, estimator, is_hardware, _ = setup_quantum_backend()
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

print(f"Complete circuit depth: {circuit.depth()}")
print(f"Total parameters: {len(circuit.parameters)}")

# Create observable for measurement
observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

# ============ HARDWARE OPTIMIZATION ============
if is_hardware:
    print("\n⚙️ Optimizing circuit for quantum hardware...")
    print(f"Backend: {backend.name}")
    print(f"Backend qubits: {backend.configuration().n_qubits}")
    print(f"Coupling map: {backend.configuration().coupling_map}")
    
    # Transpile for hardware to match the backend ISA and optimize.
    # Transpile preserves parameter expressions and produces an ISA circuit
    # compatible with the primitive validators.
    optimized_circuit = transpile(circuit, backend=backend, optimization_level=3)

    print(f"Original circuit depth: {circuit.depth()}")
    print(f"Optimized (transpiled) circuit depth: {optimized_circuit.depth()}")
    print(f"Gate count reduction: {circuit.count_ops()} → {optimized_circuit.count_ops()}")

    # Use optimized/transpiled circuit (parameterized gates preserved)
    circuit = optimized_circuit
    
    # Update observable to match the transpiled circuit layout
    # The transpiled circuit maps logical qubits to physical qubits
    # We need to map the observable accordingly
    layout = optimized_circuit.layout
    if layout is not None:
        print(f"Circuit layout mapping: {layout.final_index_layout()}")
        # Map the observable to match the physical qubits
        observable = observable.apply_layout(layout)
        print(f"Observable updated for hardware layout: {observable.num_qubits} qubits")
else:
    print("\n🔬 Using original circuit for simulation")

# Remove the duplicate observable creation line

# ============ CREATE QUANTUM NEURAL NETWORK ============
print("\n🧠 Creating Quantum Neural Network...")

# For hardware, ensure circuit is ISA-compatible after any decomposition
if is_hardware:
    # Don't decompose the transpiled circuit as it may introduce unsupported gates
    qnn_circuit = circuit
    print("Using transpiled circuit without decomposition for hardware compatibility")
else:
    # For simulation, decomposition is fine
    qnn_circuit = circuit.decompose()
    print("Using decomposed circuit for simulation")

qnn = EstimatorQNN(
    circuit=qnn_circuit,
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
)

print(f"QNN created:")
print(f"  - Input parameters: {qnn.num_inputs}")
print(f"  - Weight parameters: {qnn.num_weights}")
print(f"  - Output dimension: 1 (binary classification)")

# ============ TRAINING SETUP ============
def callback_graph(weights, obj_func_eval):
    """Enhanced callback for hardware execution monitoring"""
    objective_func_vals.append(obj_func_eval)
    iteration = len(objective_func_vals)
    
    if is_hardware:
        # For hardware, print progress without clearing output
        if iteration % 5 == 0:  # Print every 5 iterations
            print(f"  Iteration {iteration}/{MAX_ITERATIONS}: Loss = {obj_func_eval:.6f}")
            if iteration % 10 == 0:
                elapsed = time.time() - training_start_time
                avg_time = elapsed / iteration
                remaining = avg_time * (MAX_ITERATIONS - iteration)
                print(f"    Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
    else:
        # For simulation, show real-time plot
        clear_output(wait=True)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(objective_func_vals)), objective_func_vals)
        plt.show()

print(f"\n🎯 Creating classifier with {MAX_ITERATIONS} iterations...")
if is_hardware:
    print("⚠️  Hardware execution may take 30-90 minutes due to:")
    print("   - Queue waiting time")
    print("   - Circuit execution on real quantum processor")
    print("   - Error mitigation overhead")
    print(f"   - Current queue: {backend.status().pending_jobs} jobs")

classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=MAX_ITERATIONS),
    callback=callback_graph
)

# ============ TRAINING EXECUTION ============
print(f"\n{'='*60}")
print(f"🚀 STARTING QUANTUM NEURAL NETWORK TRAINING")
print(f"{'='*60}")
print(f"Backend: {backend.name}")
print(f"Dataset: {len(train_images)} training samples")
print(f"Circuit depth: {circuit.depth()}")
print(f"Parameters: {qnn.num_weights}")
print(f"Max iterations: {MAX_ITERATIONS}")

if is_hardware:
    print(f"Queue status: {backend.status().pending_jobs} jobs ahead")
    print("🔥 Starting hardware execution...")

# Prepare training data
x = np.asarray(train_images)
y = np.asarray(train_labels)

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

# Start training with timing
training_start_time = time.time()
print(f"\n⏱️ Training started at {datetime.now().strftime('%H:%M:%S')}")

classifier.fit(x, y)

training_time = time.time() - training_start_time
print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETED!")
print(f"{'='*60}")
print(f"Total training time: {training_time/60:.2f} minutes")
print(f"Iterations completed: {len(objective_func_vals)}")

# ============ EVALUATION AND RESULTS ============
# Training accuracy
train_accuracy = classifier.score(x, y)
print(f"\n📊 RESULTS:")
print(f"Training accuracy: {np.round(100 * train_accuracy, 2)}%")

# Test accuracy
y_predict = classifier.predict(test_images)
test_accuracy = classifier.score(test_images, test_labels)
print(f"Test accuracy: {np.round(100 * test_accuracy, 2)}%")

# ============ HARDWARE EXECUTION SUMMARY ============
if is_hardware:
    print(f"\n🔧 HARDWARE EXECUTION SUMMARY:")
    print(f"Backend: {backend.name}")
    print(f"Execution time: {training_time/60:.2f} minutes")
    print(f"Execution mode: Job mode (Open Plan compatible)")
    print(f"Error mitigation: Resilience level 1")

# ============ SAVE RESULTS ============
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_summary = {
    "timestamp": timestamp,
    "backend": backend.name,
    "is_hardware": is_hardware,
    "training_time_minutes": round(training_time/60, 2),
    "train_accuracy": round(100 * train_accuracy, 2),
    "test_accuracy": round(100 * test_accuracy, 2),
    "max_iterations": MAX_ITERATIONS,
    "iterations_completed": len(objective_func_vals),
    "circuit_depth": circuit.depth(),
    "num_parameters": qnn.num_weights
}

result_text = f"QNN Hardware Results - {timestamp}\n"
result_text += f"Backend: {backend.name}\n"
result_text += f"Training accuracy: {result_summary['train_accuracy']}%\n"
result_text += f"Test accuracy: {result_summary['test_accuracy']}%\n"
result_text += f"Training time: {result_summary['training_time_minutes']} minutes\n"

print(f"\n💾 SAVING RESULTS:")
print(result_text)

# Save detailed results
with open(f"qnn_hardware_results_{timestamp}.txt", "w") as f:
    f.write(result_text)

# Save JSON for analysis
import json
with open(f"qnn_hardware_results_{timestamp}.json", "w") as f:
    json.dump(result_summary, f, indent=2)

print(f"✅ Results saved to qnn_hardware_results_{timestamp}.txt/json")

# ============ PLOT TRAINING CURVE AND RESULTS ============
plt.figure(figsize=(15, 10))

# Training curve
plt.subplot(2, 3, 1)
plt.plot(objective_func_vals)
plt.title("Training Progress")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.grid(True)

# Accuracy comparison
plt.subplot(2, 3, 2)
accuracy_data = [result_summary['train_accuracy'], result_summary['test_accuracy']]
plt.bar(['Training', 'Test'], accuracy_data, color=['blue', 'orange'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, v in enumerate(accuracy_data):
    plt.text(i, v + 1, f'{v}%', ha='center')

# Sample predictions
plt.subplot(2, 3, (3, 6))
fig_pred, ax_pred = plt.subplots(2, 2, figsize=(8, 6), subplot_kw={"xticks": [], "yticks": []})
for i in range(4):
    ax_pred[i // 2, i % 2].imshow(test_images[i].reshape(2, 4), aspect="equal")
    pred_label = "Horizontal Line" if y_predict[i] == -1 else "Vertical Line"
    ax_pred[i // 2, i % 2].set_title(f"QCNN predicts: {pred_label}")
plt.subplots_adjust(wspace=0.1, hspace=0.5)

plt.tight_layout()
plt.savefig(f"qnn_hardware_results_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎯 EXECUTION COMPLETE!")
print(f"Backend used: {backend.name}")
print(f"Final test accuracy: {result_summary['test_accuracy']}%")
print(f"Results saved with timestamp: {timestamp}")
