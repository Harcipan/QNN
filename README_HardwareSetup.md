# Quantum Hardware Setup Guide

## Prerequisites

### 1. Install Required Packages
```bash
pip install qiskit-ibm-runtime
pip install qiskit-machine-learning
pip install qiskit-algorithms
```

### 2. IBM Quantum Account Setup

#### Get Your Token
1. Go to [IBM Quantum Platform](https://quantum-computing.ibm.com/)
2. Create an account or log in
3. Go to "Account" → "API Token"
4. Copy your API token

#### Save Your Token (One-time setup)
```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Replace YOUR_TOKEN_HERE with your actual token
QiskitRuntimeService.save_account(
    channel="ibm_quantum", 
    token="YOUR_TOKEN_HERE"
)
```

## Running the Code

### Option 1: Simulation Mode (Free, Fast)
```python
# In QiskitExampleRealQuantumHardware.py, set:
USE_HARDWARE = False
```

### Option 2: Real Hardware Mode
```python
# In QiskitExampleRealQuantumHardware.py, set:
USE_HARDWARE = True
HARDWARE_BACKEND = "ibm_sherbrooke"  # or your preferred backend
```

## Available Backends

### Free Backends (Open Plan)
- `ibm_brisbane` (127 qubits)
- `ibm_sherbrooke` (127 qubits) 
- `ibm_cusco` (127 qubits)
- `ibm_kyiv` (127 qubits)

### Simulator Backend
- `ibm_qasm_simulator` (free, unlimited)

## Hardware Considerations

### 🕐 Queue Times
- Free backends can have long queues (hours to days)
- Check queue status before running
- Consider running during off-peak hours

### 💰 Cost
- Free tier: Limited monthly usage
- Premium: Pay per shot/circuit execution
- Monitor your usage in IBM Quantum dashboard

### 🎯 Circuit Optimization
The code automatically:
- Reduces dataset size (20 samples vs 50)
- Reduces iterations (50 vs 200) 
- Optimizes circuit transpilation
- Adds error mitigation (resilience level 1)

### 📊 Expected Performance
- **Simulation**: ~95-100% accuracy
- **Hardware**: ~60-80% accuracy (due to noise)
- **Hardware time**: 30-90 minutes (depending on queue)

## Running Your First Hardware Job

### Step 1: Test Connection
```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print("Available backends:", [b.name for b in service.backends()])
```

### Step 2: Check Queue
```python
backend = service.backend("ibm_sherbrooke")
print(f"Queue length: {backend.status().pending_jobs}")
```

### Step 3: Run Small Test
```python
# Set small values for testing:
dataset_size = 10
max_iterations = 20
```

### Step 4: Monitor Progress
The code will show:
- Queue position
- Iteration progress  
- Estimated completion time
- Results and accuracy

## Troubleshooting

### Common Issues
1. **Token Error**: Make sure token is saved correctly
2. **Backend Unavailable**: Try different backend
3. **Long Queue**: Use simulator or try later
4. **Low Accuracy**: Normal for noisy hardware

### Error Mitigation Options
```python
# Increase resilience level for better accuracy (slower)
estimator_options.resilience_level = 2  # 0-3 scale

# Increase shots for better statistics
estimator_options.execution.shots = 4096  # default: 1024
```

## Best Practices

1. **Start Small**: Test with simulation first
2. **Check Queue**: Monitor queue before running
3. **Save Results**: Results are automatically saved to JSON
4. **Error Handling**: Code includes timeout and error recovery
5. **Session Management**: Sessions are properly closed

## File Outputs

The code generates:
- `qcnn_hardware_results_TIMESTAMP.json`: Complete results
- Console output with progress and final metrics
- Visualization plots

## Support

- [IBM Quantum Documentation](https://docs.quantum.ibm.com/)
- [Qiskit Runtime API](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime)
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)