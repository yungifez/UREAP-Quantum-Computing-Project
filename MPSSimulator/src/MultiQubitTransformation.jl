using LinearAlgebra
using TensorOperations

function cx!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = [
        0 0 1 0;
        0 0 0 1;
        1 0 0 0;
        0 1 0 0
    ]
    applyMultiQubitTransformation!(qc, qubit1, qubit2, M)
end

function cz!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = [1 0 0 0;
        0 1 0 0;
        0 0 1 0;
        0 0 0 -1]
    applyMultiQubitTransformation!(qc, qubit1, qubit2, M)
end

function cy!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = [1 0 0 0;
        0 0 0 -im;
        0 0 1 0;
        0 im 0 0]
    applyMultiQubitTransformation!(qc, qubit1, qubit2, M)
end

function swap!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = [1 0 0 0;
        0 0 1 0;
        0 1 0 0;
        0 0 0 1]
    applyMultiQubitTransformation!(qc, qubit1, qubit2, M)
end

function applyMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    if abs(qubit1 - qubit2) == 1
        applyLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    else
        applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    end
end

function applyLocalMultiQubitTransformation!(qc::QuantumCircuit, qubitIndex1::Int, qubitIndex2::Int, twoQubitGate::Matrix)
    siteTensor1 = qc.state[qubitIndex1]
    siteTensor2 = qc.state[qubitIndex2]

    # Reshape to (out1, out2, in1, in2)
    gateTensor = reshape(ComplexF64.(twoQubitGate), 2, 2, 2, 2)
    contractedTensor = getStatevectorFromTensorTrain(VecOrMat{ComplexF64}[siteTensor1, siteTensor2])

    if qubitIndex1 == 1
        # contractedTensor: [op1, op2, r]
        @tensor transformedTensor[op1, op2, r] :=
            gateTensor[p1, op1, p2, op2] * contractedTensor[p1, p2, r]

        # Split: (p1) and (p2, r)
        p1, p2, r = size(transformedTensor)
        U, S, Vt = svd(reshape(transformedTensor, p1, p2 * r))

    elseif qubitIndex1 == length(qc.state) - 1
        @tensor transformedTensor[l, op1, op2] :=
            gateTensor[p1, op1, p2, op2] * contractedTensor[l, p1, p2]

        # Split: (l, p1) and (p2)
        l, p1, p2 = size(transformedTensor)
        U, S, Vt = svd(reshape(transformedTensor, l * p1, p2))
    else
        # contractedTensor: [l, op1, op2, r]
        @tensor transformedTensor[l, op1, op2, r] :=
            gateTensor[p1, op1, p2, op2] * contractedTensor[l, p1, p2, r]

        # Split: (l, p1) and (p2, r)
        l, p1, p2, r = size(transformedTensor)
        U, S, Vt = svd(reshape(transformedTensor, l * p1, p2 * r))
    end

    qc.state[qubitIndex1] = U * Diagonal(S)
    qc.state[qubitIndex2] = Matrix{ComplexF64}(Vt)
end

function applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
end

