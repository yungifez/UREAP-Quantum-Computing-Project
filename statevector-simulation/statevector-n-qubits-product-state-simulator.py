from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from memory_profiler import memory_usage
import json
import os
import sys


def run_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)

    qc.measure_all()

    sampler = StatevectorSampler()
    sampler.run([qc], shots=1024).result()


def profile_qubits(n_qubits):
    return memory_usage(
        (run_circuit, (n_qubits,), {}),
        max_usage=True
    )


if __name__ == "__main__":
    n = int(sys.argv[1])

    mem = profile_qubits(n)
    print(f"{n} qubits: {mem:.2f} MiB")

    filename = "product-state-memory.json"

    if os.path.exists(filename):
        with open(filename, "r") as f:
            all_results = json.load(f)
        all_results = {int(k): v for k, v in all_results.items()}
    else:
        all_results = {}

    all_results[n] = mem

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
