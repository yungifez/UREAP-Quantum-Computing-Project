using LinearAlgebra
using TensorOperations

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
            # Each tensor is (2 x 1 x 1) flattened into matrix form
            A = zeros(2, 1)

            # Set |0⟩ amplitude
            A[1, 1] = 1.0 + 0.0im

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

function getStatevectorFromTensorTrain(list)
    n = length(list)
    state = list[1]

    for i in 2:n
        curr = list[i]
        bond_dim = size(state, ndims(state))

        flat_state = reshape(state, :, bond_dim)
        flat_curr = reshape(curr, bond_dim, :)
        contracted = flat_state * flat_curr

        if i == n
            num_qubits = round(Int, log2(length(contracted)))
            state = reshape(contracted, ntuple(_ -> 2, num_qubits))
        else
            r_out = size(curr, ndims(curr))
            num_phys_so_far = round(Int, log2(length(contracted) ÷ r_out))
            state = reshape(contracted, (ntuple(_ -> 2, num_phys_so_far)..., r_out))
        end
    end
    return state
end
