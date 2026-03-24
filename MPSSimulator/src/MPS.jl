using LinearAlgebra

using Base: IdentityUnitRange
struct QuantumCircuit
    state::Vector{VecOrMat{ComplexF64}}
    # maxBondDimension::Int

    # We need to prepare an initial state that would represent a |000.......>
    # we would do true approximations for now
    #
    # Not sure why this works but test runs correctly lol
    function QuantumCircuit(n::Int)
        state = Vector{Matrix{ComplexF64}}()

        for i in 1:n
            previousBondDimension = min(2^(i - 1), 2^(n - (i - 1)))
            currentBondDimeonsion = min(2^i, 2^(n - i))

            A = zeros(previousBondDimension * 2, currentBondDimeonsion)
            A[diagind(A)] .= 1

            push!(state, A)
        end

        new(state)
    end
end

function reshapeTensorForMPS(tensor::AbstractArray)::Matrix
    if all(size(tensor) .!= 2)
        throw(ArgumentError("Tensor must have shape (2,2,2,2.....), but got $(size(tensor))"))
    end

    n = ndims(tensor)
    return reshape(tensor, size(tensor, 1), 2^(n - 1))
end

function createTensorTrainFromReshapedArray(matrix::Matrix)::Vector{VecOrMat{ComplexF64}}
    "Recursive function to compute MPS from original shape"

    if size(matrix, 2) == 1
        return [ComplexF64.(matrix)]
    end

    A = svd(matrix)

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
    rowSize, colSize = size(sV)

    reshapedSV = reshape(sV, 2 * rowSize, :)

    return [A.U, createTensorTrainFromReshapedArray(reshapedSV)...]
end

function getStatevectorFromTensorTrain(list::Vector{VecOrMat{ComplexF64}})
    n = length(list)
    state = list[1]

    for i in 2:n
        current = list[i]
        stateRowSize, stateColSize = size(state)
        currRowSize, currColSize = size(current)

        # Reshape to match the dimensions of the previous matrix
        current = reshape(current, trunc(Int, currRowSize / 2), :)
        state = reshape(state, :, trunc(Int, currRowSize / 2))
        state = state * current
    end

    total_elements = length(state)
    n_qubits = Int(log2(total_elements))
    dynamic_shape = ntuple(_ -> 2, n_qubits)

    return reshape(state, dynamic_shape)
end

