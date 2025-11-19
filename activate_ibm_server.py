from qiskit_ibm_runtime import QiskitRuntimeService
 
QiskitRuntimeService.save_account(
    token="M8RcoQN0IaIBt1JH-ivZZhKzmU1f7VXTL59FOt4iaXZQ", 
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