import json
import matplotlib.pyplot as plt

qubits = []
memory = []

with open("product-state-memory.json") as f:
    data = json.load(f)

for q in sorted(data, key=lambda x: int(x)):
    qubits.append(int(q))
    memory.append(float(data[q]))

plt.figure()
plt.plot(qubits, memory, marker='o')
plt.xlabel("Number of Qubits")
plt.ylabel("Peak Memory Usage (MiB)")
plt.title("Product State Statevector Memory Growth")
plt.grid(True)
plt.show()
