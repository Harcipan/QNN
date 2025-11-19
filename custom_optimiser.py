import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA, Optimizer, OptimizerResult, OptimizerSupportLevel
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from typing import Optional, Callable, Union, List
import time

algorithm_globals.random_seed = 12345
estimator = Estimator()


class CustomOptimizer(Optimizer):
    """
    A custom gradient-based optimizer with momentum and adaptive learning rate.
    
    This optimizer implements a basic gradient descent with momentum and can be
    easily extended with additional features like adaptive learning rates,
    different momentum strategies, etc.
    """
    
    def __init__(
        self,
        maxiter: int = 1000,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        tolerance: float = 1e-6,
        patience: int = 20,
        adaptive_lr: bool = True,
        lr_decay: float = 0.95,
        finite_diff_step: float = 1e-6
    ):
        """
        Args:
            maxiter: Maximum number of iterations
            learning_rate: Initial learning rate
            momentum: Momentum coefficient (0 = no momentum, 1 = full momentum)
            tolerance: Convergence tolerance
            patience: Number of iterations to wait for improvement before early stopping
            adaptive_lr: Whether to use adaptive learning rate
            lr_decay: Learning rate decay factor
            finite_diff_step: Step size for finite difference gradient approximation
        """
        super().__init__()
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.tolerance = tolerance
        self.patience = patience
        self.adaptive_lr = adaptive_lr
        self.lr_decay = lr_decay
        self.finite_diff_step = finite_diff_step
        
        # Internal state
        self._velocity = None
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._iteration = 0
        
    def minimize(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bounds: Optional[List] = None
    ) -> OptimizerResult:
        """
        Minimize the objective function.
        
        Args:
            fun: Objective function to minimize
            x0: Initial parameter values
            jac: Gradient function (if None, will use finite differences)
            bounds: Parameter bounds (not implemented in this basic version)
            
        Returns:
            OptimizerResult containing optimization results
        """
        start_time = time.time()
        
        # Initialize
        x = np.copy(x0)
        self._velocity = np.zeros_like(x)
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._iteration = 0
        
        # History tracking
        history = {
            'loss': [],
            'parameters': [],
            'learning_rates': []
        }
        
        # Main optimization loop
        for iteration in range(self.maxiter):
            self._iteration = iteration
            
            # Evaluate objective function
            current_loss = fun(x)
            history['loss'].append(current_loss)
            history['parameters'].append(x.copy())
            history['learning_rates'].append(self.learning_rate)
            
            # Check for improvement
            if current_loss < self._best_loss - self.tolerance:
                self._best_loss = current_loss
                self._patience_counter = 0
            else:
                self._patience_counter += 1
            
            # Early stopping
            if self._patience_counter >= self.patience:
                print(f"Early stopping at iteration {iteration} (no improvement for {self.patience} iterations)")
                break
            
            # Compute gradient
            if jac is not None:
                gradient = jac(x)
            else:
                gradient = self._finite_difference_gradient(fun, x)
            
            # Update velocity (momentum)
            self._velocity = self.momentum * self._velocity - self.learning_rate * gradient
            
            # Update parameters
            x_new = x + self._velocity
            
            # Simple bounds handling (clip to [-2π, 2π] for quantum parameters)
            if bounds is None:
                x_new = np.clip(x_new, -2*np.pi, 2*np.pi)
            
            x = x_new
            
            # Adaptive learning rate
            if self.adaptive_lr and iteration > 0:
                # Decay learning rate if no improvement
                if self._patience_counter > 5:
                    self.learning_rate *= self.lr_decay
                # Reset learning rate occasionally
                if iteration % 50 == 0:
                    self.learning_rate = self.initial_lr * (0.95 ** (iteration // 50))
            
            # Progress reporting
            if iteration % 10 == 0 or iteration < 10:
                print(f"Iteration {iteration}: Loss = {current_loss:.6f}, LR = {self.learning_rate:.6f}")
        
        # Final evaluation
        final_loss = fun(x)
        total_time = time.time() - start_time
        
        # Create result
        result = OptimizerResult()
        result.x = x
        result.fun = final_loss
        result.nit = self._iteration + 1
        result.success = self._patience_counter < self.patience
        result.message = f"Optimization completed in {self._iteration + 1} iterations"
        
        # Store additional information
        result.history = history
        result.total_time = total_time
        
        print(f"Optimization completed: Final loss = {final_loss:.6f} in {total_time:.2f} seconds")
        
        return result
    
    def _finite_difference_gradient(self, fun: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient using finite differences.
        """
        gradient = np.zeros_like(x)
        fx = fun(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += self.finite_diff_step
            
            x_minus = x.copy()
            x_minus[i] -= self.finite_diff_step
            
            # Central difference
            gradient[i] = (fun(x_plus) - fun(x_minus)) / (2 * self.finite_diff_step)
        
        return gradient
    
    @property
    def settings(self) -> dict:
        """Return optimizer settings."""
        return {
            'maxiter': self.maxiter,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'tolerance': self.tolerance,
            'patience': self.patience,
            'adaptive_lr': self.adaptive_lr,
            'lr_decay': self.lr_decay,
            'finite_diff_step': self.finite_diff_step
        }
    
    def get_support_level(self):
        """Return support level dictionary."""
        return {
            'gradient': OptimizerSupportLevel.ignored,
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.required
        }


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
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    
    print(f"Objective function value: {obj_func_eval}")
    # # Create or update the plot without blocking
    # plt.clf()  # Clear the current figure
    # plt.title("Objective function value against iteration")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective function value")
    # plt.plot(range(len(objective_func_vals)), objective_func_vals)
    # plt.draw()
    # plt.pause(0.00001)  # Very small pause to update without blocking

# classifier = NeuralNetworkClassifier(
#     qnn,
#     optimizer=CustomOptimizer(maxiter=2, learning_rate=0.01, momentum=0.9),  # Using custom optimizer
#     callback=callback_graph
# )
    
# x = np.asarray(train_images)
# y = np.asarray(train_labels)

objective_func_vals = []
# plt.ion()  # Turn on interactive mode
# plt.rcParams["figure.figsize"] = (12, 6)
# classifier.fit(x, y)

# # score classifier
# print(f"Accuracy from the train data : {np.round(100 * classifier.score(x, y), 2)}%")

# y_predict = classifier.predict(test_images)
# x = np.asarray(test_images)
# y = np.asarray(test_labels)
# print(f"Accuracy from the test data : {np.round(100 * classifier.score(x, y), 2)}%")

# # Let's see some examples in our dataset
# fig, ax = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={"xticks": [], "yticks": []})
# for i in range(0, 4):
#     ax[i // 2, i % 2].imshow(test_images[i].reshape(2, 4), aspect="equal")
#     if y_predict[i] == -1:
#         ax[i // 2, i % 2].set_title("The QCNN predicts this is a Horizontal Line")
#     if y_predict[i] == +1:
#         ax[i // 2, i % 2].set_title("The QCNN predicts this is a Vertical Line")
# plt.subplots_adjust(wspace=0.1, hspace=0.5)
# plt.show(block=False)  # Non-blocking show
# plt.pause(0.1)  # Brief pause to render the plot


def compare_optimizers():
    """
    Function to compare different optimizers on the same problem.
    You can call this to benchmark your custom optimizer against COBYLA.
    """
    print("Comparing optimizers...")
    
    # Test with COBYLA
    print("\n=== Testing COBYLA ===")
    try:
        classifier_cobyla = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=10),  # Increased iterations for COBYLA
            callback=None  # Disable callback for cleaner output
        )
        
        start_time = time.time()
        print("Starting COBYLA training...")
        classifier_cobyla.fit(np.asarray(train_images), np.asarray(train_labels))
        cobyla_time = time.time() - start_time
        print(f"COBYLA training completed in {cobyla_time:.2f} seconds")
        
        cobyla_train_acc = classifier_cobyla.score(np.asarray(train_images), np.asarray(train_labels))
        cobyla_test_acc = classifier_cobyla.score(np.asarray(test_images), np.asarray(test_labels))
        cobyla_success = True
        
    except Exception as e:
        print(f"COBYLA failed with error: {type(e).__name__}: {e}")
        cobyla_time = float('inf')
        cobyla_train_acc = 0.0
        cobyla_test_acc = 0.0
        cobyla_success = False
    
    # Test with Custom Optimizer
    print("\n=== Testing Custom Optimizer ===")
    try:
        classifier_custom = NeuralNetworkClassifier(
            qnn,
            optimizer=CustomOptimizer(maxiter=1, learning_rate=0.01, momentum=0.9),  # Matched iterations
            callback=None  # Disable callback for cleaner output
        )
        
        start_time = time.time()
        print("Starting Custom optimizer training...")
        classifier_custom.fit(np.asarray(train_images), np.asarray(train_labels))
        custom_time = time.time() - start_time
        print(f"Custom optimizer training completed in {custom_time:.2f} seconds")
        
        custom_train_acc = classifier_custom.score(np.asarray(train_images), np.asarray(train_labels))
        custom_test_acc = classifier_custom.score(np.asarray(test_images), np.asarray(test_labels))
        custom_success = True
        
    except Exception as e:
        print(f"Custom optimizer failed with error: {type(e).__name__}: {e}")
        custom_time = float('inf')
        custom_train_acc = 0.0
        custom_test_acc = 0.0
        custom_success = False
    
    # Print comparison results
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<25} {'COBYLA':<15} {'Custom':<15}")
    print("-"*60)
    print(f"{'Success':<25} {'Yes' if cobyla_success else 'No':<15} {'Yes' if custom_success else 'No':<15}")
    if cobyla_success and custom_success:
        print(f"{'Training Time (s)':<25} {cobyla_time:<15.2f} {custom_time:<15.2f}")
        print(f"{'Train Accuracy (%)':<25} {100*cobyla_train_acc:<15.2f} {100*custom_train_acc:<15.2f}")
        print(f"{'Test Accuracy (%)':<25} {100*cobyla_test_acc:<15.2f} {100*custom_test_acc:<15.2f}")
        
        # Analysis
        print(f"{'Speed Improvement':<25} {'-':<15} {f'{cobyla_time/custom_time:.2f}x' if custom_time > 0 else 'N/A':<15}")
    elif custom_success:
        print(f"{'Training Time (s)':<25} {'Failed':<15} {custom_time:<15.2f}")
        print(f"{'Train Accuracy (%)':<25} {'Failed':<15} {100*custom_train_acc:<15.2f}")
        print(f"{'Test Accuracy (%)':<25} {'Failed':<15} {100*custom_test_acc:<15.2f}")
    print("="*60)
    
    return {
        'cobyla': {'time': cobyla_time, 'train_acc': cobyla_train_acc, 'test_acc': cobyla_test_acc, 'success': cobyla_success},
        'custom': {'time': custom_time, 'train_acc': custom_train_acc, 'test_acc': custom_test_acc, 'success': custom_success}
    }


# Uncomment the line below to run the comparison
comparison_results = compare_optimizers()