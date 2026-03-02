using LinearAlgebra

function cx!(qc::QuantumCircuit, qubit1::Int, qubit2::Int)
    M = [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
    applyMultiQubitTransformation!(qc, qubit1, qubit2, M)
end

function applyMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    if abs(qubit1 - qubit2) == 1
        applyLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    else
        applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    end
end

function applyLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
    # TODO test
    site1 = qc.state[min(qubit1, qubit2)]
    site2 = qc.state[max(qubit1, qubit2)]

    site1RowSize, site1ColSize = size(site1)
    site2RowSize, site2ColSize = size(site2)

    site2 = reshape(site2, trunc(Int, site2RowSize / 2), site2ColSize * 2)

    contracted = site1 * site2
    transformationRowSize, transformationColSize = size(transformation)
    contracted = reshape(contracted, transformationRowSize, :)

    # Order matters here T(x) = Ax
    Tx = transformation * contracted
    display(Tx)

    # Seperate the two sites back to original
    A = svd(Tx)

    # Stabilize the sign of each column of U
    # choosing a deterministic convention to prevent scaling eiganvalue issues
    # Doesnt change state, just makes sure local sites are positive
    for i in 1:size(A.U, 2)
        firstNonZero = findfirst(x -> abs(x) > 1e-12, A.U[:, i])

        if firstNonZero !== nothing
            phase = A.U[firstNonZero, i] / abs(A.U[firstNonZero, i])
            A.U[:, i] .= A.U[:, i] / phase
            A.Vt[i, :] .= A.Vt[i, :] * phase
        end
    end

    sV = Diagonal(A.S) * A.Vt

    reshapedSV = reshape(sV, site1RowSize, site1ColSize)

    qc.state[qubit1] = reshapedSV
    qc.state[qubit2] = A.U

    # site2 = reshape(site2, site2RowSize, site2ColSize)
    # display(A.U)
    # display(reshapedSV)
end

function applyNonLocalMultiQubitTransformation!(qc::QuantumCircuit, qubit1::Int, qubit2::Int, transformation::Matrix)
end

