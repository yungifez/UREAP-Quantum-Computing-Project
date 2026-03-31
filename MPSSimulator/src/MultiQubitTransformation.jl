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
    reverseGateContraction = false
    if q1 > q2
        q1, q2 = q2, q1
        reverseGateContraction = true
    end

    A = reshape(qc.state[q1], 2, div(size(qc.state[q1], 1), 2), :)
    B = reshape(qc.state[q2], 2, div(size(qc.state[q2], 1), 2), :)

    G = reshape(ComplexF64.(gate), 2, 2, 2, 2)
    @tensor combined[p1, p2, l, r] := A[p1, l, b] * B[p2, b, r]

    if !reverseGateContraction
        @tensor transformed[o1, o2, l, r] := G[i1, o1, i2, o2] * combined[i1, i2, l, r]
    else
        @tensor transformed[o1, o2, l, r] := G[i1, o1, i2, o2] * combined[i1, i2, l, r]
    end

    T = permutedims(transformed, (3, 1, 2, 4))

    dl, do1, do2, dr = size(T)

    mat = reshape(T, dl * do1, do2 * dr)

    U, S, Vt = svd(mat)

    χ = length(S)

    US = U * Diagonal(S)

    U_tensor = reshape(US, dl, do1, χ)      # (l, o1, χ)
    V_tensor = reshape(Vt, χ, do2, dr)      # (χ, o2, r)

    A = permutedims(U_tensor, (2, 1, 3))  # (o1, l, χ)
    B = permutedims(V_tensor, (2, 1, 3))  # (o2, χ, r)

    qc.state[q1] = reshape(A, :, size(A, 3))
    qc.state[q2] = reshape(B, :, size(B, 3))
end

function applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
end

