for q in 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30; do
    python statevector-n-qubits-entanglement-state-simulator.py $q
done

mkdir -p logs
cp product-state-memory.json logs/product-state-memory-$(date +%Y-%m-%d_%H-%M-%S).json

