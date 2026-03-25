using LinearAlgebra
using TensorOperations

function cx!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = [
        1 0 0 0;
        0 1 0 0;
        0 0 0 1;
        0 0 1 0
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

function applyLocalMultiQubitTransformation!(qc::QuantumCircuit, q1::Int, q2::Int, gate::Matrix)
    if q1 > q2
        q1, q2 = q2, q1
    end

    A = reshape(qc.state[q1], 2, 1, :)

    bond1 = size(qc.state[q1], 2)
    B = reshape(qc.state[q2], 2, bond1, :)

    G = reshape(ComplexF64.(gate), 2, 2, 2, 2)

    println(size(A))
    println(size(B))

    @tensor combined[p1, p2, l, r] := A[p1, l, b] * B[p2, b, r]

    @tensor transformed[o1, o2, l, r] := G[o1, o2, i1, i2] * combined[i1, i2, l, r]

    L, R = size(A, 2), size(B, 3)
    U, S, Vt = svd(reshape(transformed, 2 * L, 2 * R))

    new_bond = length(S)
    US = U * Diagonal(S)

    qc.state[q1] = reshape(US, 2, new_bond)
    qc.state[q2] = reshape(Matrix{ComplexF64}(Vt), 2 * new_bond, R)
end

function applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
end

