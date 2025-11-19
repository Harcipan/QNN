import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RYGate, RZGate, CXGate

def improved_conv_circuit(params):
    """
    Enhanced convolution circuit with more expressivity.
    Uses 6 parameters instead of 3 for richer parameterization.
    """
    target = QuantumCircuit(2)
    
    # Initial single qubit rotations
    target.ry(params[0], 0)
    target.ry(params[1], 1)
    
    # Entangling gate
    target.cx(0, 1)
    
    # More rotations after entanglement
    target.ry(params[2], 0)
    target.ry(params[3], 1)
    target.rz(params[4], 0)
    target.rz(params[5], 1)
    
    # Second entangling gate for more expressivity
    target.cx(1, 0)
    
    return target


def hardware_efficient_conv_circuit(params):
    """
    Hardware-efficient ansatz style convolution.
    Good for NISQ devices with limited connectivity.
    """
    target = QuantumCircuit(2)
    
    # Layer 1: Single qubit rotations
    target.ry(params[0], 0)
    target.ry(params[1], 1)
    target.rz(params[2], 0)
    target.rz(params[3], 1)
    
    # Entangling layer
    target.cx(0, 1)
    
    # Layer 2: More single qubit rotations
    target.ry(params[4], 0)
    target.ry(params[5], 1)
    
    return target


def data_reuploading_conv_circuit(params, data=None):
    """
    Data re-uploading convolution circuit.
    Encodes data multiple times throughout the circuit.
    """
    target = QuantumCircuit(2)
    
    if data is not None:
        # First data encoding
        target.ry(data[0], 0)
        target.ry(data[1], 1)
    
    # Parameterized gates
    target.ry(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    
    if data is not None:
        # Re-upload data
        target.ry(data[0] * params[2], 0)
        target.ry(data[1] * params[3], 1)
    
    # More parameterized gates
    target.ry(params[4], 0)
    target.ry(params[5], 1)
    target.cx(1, 0)
    
    return target


def improved_pool_circuit(params):
    """
    Enhanced pooling circuit with controlled rotations.
    Uses the first qubit to control operations on the second.
    """
    target = QuantumCircuit(2)
    
    # Controlled rotations - more sophisticated pooling
    target.cry(params[0], 0, 1)  # Controlled RY
    target.crz(params[1], 0, 1)  # Controlled RZ
    
    # Mix the qubits
    target.cx(0, 1)
    
    # More controlled operations
    target.cry(params[2], 1, 0)
    
    # Final single qubit rotation on the "pooled" qubit
    target.ry(params[3], 1)
    
    return target


def measurement_based_pool_circuit(params):
    """
    Measurement-based pooling (conceptual - would need adaptation for actual use).
    The idea is to perform a partial measurement to extract information.
    """
    target = QuantumCircuit(2)
    
    # Prepare entangled state
    target.ry(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    
    # Conditional operations based on qubit states
    target.cry(params[2], 0, 1)
    target.crx(params[3], 1, 0)
    
    return target


def improved_conv_layer(num_qubits, param_prefix, layer_type="improved"):
    """
    Enhanced convolutional layer with different circuit types.
    Similar structure to original but with improved circuits.
    """
    qc = QuantumCircuit(num_qubits, name=f"Enhanced Conv Layer ({layer_type})")
    qubits = list(range(num_qubits))
    param_index = 0
    
    # Choose circuit type and parameter count
    if layer_type == "improved":
        params_per_gate = 6
        circuit_func = improved_conv_circuit
    elif layer_type == "hardware_efficient":
        params_per_gate = 6
        circuit_func = hardware_efficient_conv_circuit
    else:  # data_reuploading
        params_per_gate = 6
        circuit_func = data_reuploading_conv_circuit
    
    # Use same structure as original conv_layer but with more parameters
    # Calculate total parameters: same number of gates as original but more params per gate
    total_gates = num_qubits  # This matches the original conv_layer parameter calculation
    params = ParameterVector(param_prefix, length=total_gates * params_per_gate)
    
    # First round: even pairs (0,1), (2,3), (4,5), ...
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        circuit_params = params[param_index:param_index + params_per_gate]
        qc = qc.compose(circuit_func(circuit_params), [q1, q2])
        qc.barrier()
        param_index += params_per_gate
    
    # Second round: odd pairs (1,2), (3,4), (5,6), ... + wrap around
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [qubits[0]]):
        if param_index + params_per_gate <= len(params):
            circuit_params = params[param_index:param_index + params_per_gate]
            qc = qc.compose(circuit_func(circuit_params), [q1, q2])
            qc.barrier()
            param_index += params_per_gate
    
    qc_inst = qc.to_instruction()
    
    qc_final = QuantumCircuit(num_qubits)
    qc_final.append(qc_inst, qubits)
    return qc_final


def improved_pool_layer(sources, sinks, param_prefix, pool_type="improved"):
    """
    Enhanced pooling layer with different pooling strategies.
    Same structure as original but with improved circuits.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name=f"Enhanced Pool Layer ({pool_type})")
    
    # Choose pooling circuit type
    if pool_type == "improved":
        params_per_gate = 4
        circuit_func = improved_pool_circuit
    else:  # measurement_based
        params_per_gate = 4
        circuit_func = measurement_based_pool_circuit
    
    params = ParameterVector(param_prefix, length=len(sources) * params_per_gate)
    param_index = 0
    
    for source, sink in zip(sources, sinks):
        circuit_params = params[param_index:param_index + params_per_gate]
        qc = qc.compose(circuit_func(circuit_params), [source, sink])
        qc.barrier()
        param_index += params_per_gate
    
    qc_inst = qc.to_instruction()
    
    qc_final = QuantumCircuit(num_qubits)
    qc_final.append(qc_inst, range(num_qubits))
    return qc_final


def residual_conv_layer(num_qubits, param_prefix):
    """
    Residual convolution layer inspired by ResNets.
    Adds skip connections to help with gradient flow.
    """
    qc = QuantumCircuit(num_qubits, name="Residual Conv Layer")
    
    # Store initial state (conceptually - in practice we apply identity)
    qc.barrier(label="Residual Start")
    
    # Apply convolution
    conv_layer = improved_conv_layer(num_qubits, param_prefix + "_conv", "hardware_efficient")
    qc = qc.compose(conv_layer)
    
    # Skip connection: add small rotation based on "residual"
    residual_params = ParameterVector(param_prefix + "_residual", length=num_qubits)
    for i in range(num_qubits):
        qc.ry(residual_params[i] * 0.1, i)  # Small residual correction
    
    qc.barrier(label="Residual End")
    return qc


def attention_conv_layer(num_qubits, param_prefix):
    """
    Attention-inspired convolution layer.
    Uses controlled operations to create attention-like mechanisms.
    """
    qc = QuantumCircuit(num_qubits, name="Attention Conv Layer")
    
    # Attention weights (which qubits to pay attention to)
    attention_params = ParameterVector(param_prefix + "_attention", length=num_qubits * 2)
    
    # Create "attention" by applying controlled rotations
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                # Attention from qubit i to qubit j
                param_idx = i * 2
                qc.cry(attention_params[param_idx], i, j)
                qc.crz(attention_params[param_idx + 1], i, j)
    
    qc.barrier(label="Attention")
    
    # Standard convolution after attention
    conv_layer = improved_conv_layer(num_qubits, param_prefix + "_conv", "improved")
    qc = qc.compose(conv_layer)
    
    return qc


# Example usage and comparison function
def compare_layer_types():
    """
    Compare different layer types for expressivity and parameter count.
    """
    num_qubits = 4
    
    print("=== LAYER COMPARISON ===\n")
    
    # Original layers
    from test_improved_ansatz import conv_layer, pool_layer
    
    try:
        original_conv = conv_layer(num_qubits, "orig")
        original_params = len(original_conv.parameters)
        print(f"Original Conv Layer: {original_params} parameters")
    except:
        print("Original conv_layer not available")
    
    # Improved layers
    improved_conv = improved_conv_layer(num_qubits, "imp", "improved")
    hw_eff_conv = improved_conv_layer(num_qubits, "hw", "hardware_efficient") 
    residual_conv = residual_conv_layer(num_qubits, "res")
    attention_conv = attention_conv_layer(num_qubits, "att")
    
    print(f"Improved Conv Layer: {len(improved_conv.parameters)} parameters")
    print(f"Hardware Efficient Conv: {len(hw_eff_conv.parameters)} parameters")
    print(f"Residual Conv Layer: {len(residual_conv.parameters)} parameters")
    print(f"Attention Conv Layer: {len(attention_conv.parameters)} parameters")
    
    print(f"\nCircuit Depths:")
    print(f"Improved Conv: {improved_conv.depth()}")
    print(f"Hardware Efficient: {hw_eff_conv.depth()}")
    print(f"Residual Conv: {residual_conv.depth()}")
    print(f"Attention Conv: {attention_conv.depth()}")
    
    return {
        'improved': improved_conv,
        'hardware_efficient': hw_eff_conv,
        'residual': residual_conv,
        'attention': attention_conv
    }


if __name__ == "__main__":
    # Run comparison
    layers = compare_layer_types()
    
    # Draw one example
    print("\n=== Example: Improved Conv Layer ===")
    print(layers['improved'].draw())