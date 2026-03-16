using LinearAlgebra
using TensorOperations

function cx!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = ComplexF64[
        1 0 0 0;
        0 1 0 0;
        0 0 0 1; # Row 3: maps |10> to |11>
        0 0 1 0  # Row 4: maps |11> to |10>
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
    site1 = qc.state[qubitIndex1]
    site2 = qc.state[qubitIndex2]

    l1, r1 = size(site1)
    l2, r2 = size(site2)
    l1 ÷= 2
    l2 ÷= 2

    A = reshape(site1, l1, 2, r1)
    B = reshape(site2, l2, 2, r2)

    @tensor combined[ll, i1, i2, rr] := A[ll, i1, b] * B[b, i2, rr]

    gateTensor = reshape(ComplexF64.(twoQubitGate), 2, 2, 2, 2)

    @tensor transformed[ll, o1, o2, rr] := gateTensor[o1, o2, i1, i2] * combined[ll, i1, i2, rr]

    U, S, Vt = svd(reshape(transformed, l1 * 2, 2 * r2))
    χ = length(S)

    qc.state[qubitIndex1] = U
    qc.state[qubitIndex2] = reshape(Diagonal(S) * Vt, χ * 2, r2)
end

function applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
end

