using Test, LinearAlgebra, Random

using ..MPSSimulator

function printListOfArrays(list)
    n = length(list)
    for i in 1:n
        display(list[i])
    end
end

@testset "reshape tensor works on 4d tensor" begin
    # 4D tensor
    A = reshape(collect(1:16), 2, 2, 2, 2)  # shape: (2,2,2,2)

    # Apply reshape
    B = reshapeTensorForMPS(A)

    # This looks weird but Julia is a column major language so it means that
    # it stores elements column wise ie [1,3; 2,4], looks like
    # | 1 2 |
    # | 3 4 |
    # tbis is just a 2d slice of the 4d tensor
    # The first 3d slice looks something like
    # | 1 2 |
    # | 3 4 |
    # --------
    # | 5 7 |
    # | 6 8 |
    # 
    # And the full thing looks something like
    # \\\\\\\\\\\\\\
    # | 1 2 |      |
    # | 3 4 |      |
    # --------     |
    # | 5 7 |      |
    # | 6 8 |      |
    # \\\\\\\\\\\\\\
    # | 9 11 |      |
    # | 10 12 |      |
    # --------     |
    # | 13 15 |      |
    # | 14 16 |      |
    # \\\\\\\\\\\\\
    @test size(B) == (2, 8)        # Check matrix shap

    # When we reshape, we make the first 2 indices the rows and the remaining indices are collapsed into columns
    # Therefore, the first column slice corresponds to all rows in the first column across all pages and blocks:
    #   T[1,1,1,1] = 1   # row 1, column 1, page 1, block 1
    #   T[2,1,1,1] = 2   # row 2, column 1, page 1, block 1
    # Then the second column slice corresponds to all rows in the second column across all pages and blocks:
    #   T[1,2,1,1] = 3   # row 1, column 2, page 1, block 1
    #   T[2,2,1,1] = 4   # row 2, column 2, page 1, block 1
    # Continuing in this pattern across all pages and blocks, we get the reshaped 2×8 matrix:
    # [1  3  5  7  9 11 13 15;
    #  2  4  6  8 10 12 14 16]
    @test B[1, 1] == 1                    # First row
    @test B[2, 1] == 2                    # second element
    @test A[2, 1, 1, 2] == B[2, 5]        # tenth element
    @test A[2, 2, 2, 2] == B[2, 8]        # 16th element
end


@testset "reshape tensor works for random n-dimensional tensor" begin
    n = rand(1:10)
    shape = ntuple(_ -> 2, n)
    N = 2^n
    A = reshape(collect(1:N), shape...)

    B = reshapeTensorForMPS(A)

    @test size(B) == (2, div(N, 2))
    @test B[ntuple(_ -> 1, n)...] == 1
    @test B[2, ntuple(_ -> 1, n - 1)...] == 2
    # TODO test the other items are where they are meant to be
    # @test B[ntuple(_ -> 2, n)...] == N
end


@testset "MPS Construction: n-Qubit GHZ State" begin
    n = rand(1:10)
    N = 2^n
    ghz_vec = zeros(ComplexF64, N)
    @info("N = $n for the test")
    ghz_vec[1] = 1.0 / sqrt(2)
    ghz_vec[end] = 1.0 / sqrt(2)

    A = reshape(ghz_vec, ntuple(_ -> 2, n))

    B = reshapeTensorForMPS(A)

    C = createTensorTrainFromReshapedArray(B)

    @test length(C) == n

    for i in 1:n
        previousBondDimension = min(2^(i - 1), 2^(n - (i - 1)))
        currentBondDimeonsion = min(2^i, 2^(n - i))
        @test size(C[i]) == (previousBondDimension * 2, currentBondDimeonsion)
    end
end

@testset "Test we can get back exact statevector for GHZ state" begin
    n = rand(1:10)
    N = 2^n
    ghz_vec = zeros(ComplexF64, N)
    @info("N = $n for the test")
    ghz_vec[1] = 1.0 / sqrt(2)
    ghz_vec[end] = 1.0 / sqrt(2)

    A = reshape(ghz_vec, ntuple(_ -> 2, n))

    B = reshapeTensorForMPS(A)

    C = createTensorTrainFromReshapedArray(B)

    D = getStatevectorFromTensorTrain(C)

    @test D ≈ A atol = 1e-12
end

@testset "Test we can get back exact statevector for entangled state" begin
    n = rand(2:10)
    N = 2^n
    @info("Number of qubits: $n")

    Random.seed!(1234)
    ent_vec = randn(ComplexF64, N) + im * randn(ComplexF64, N)
    ent_vec /= norm(ent_vec)  # normalize

    A = reshape(ent_vec, ntuple(_ -> 2, n))

    B = reshapeTensorForMPS(A)
    C = createTensorTrainFromReshapedArray(B)

    D = getStatevectorFromTensorTrain(C)

    @test D ≈ A atol = 1e-12
    @test norm(D) ≈ 1 atol = 1e-12
end

@testset "Test we can get back exact statevector for zeroed state" begin
    n = rand(1:10)
    N = 2^n
    zero_vec = zeros(ComplexF64, N)
    @info("N = $n for the test")

    A = reshape(zero_vec, ntuple(_ -> 2, n))

    B = reshapeTensorForMPS(A)

    C = createTensorTrainFromReshapedArray(B)

    D = getStatevectorFromTensorTrain(C)

    @test D ≈ A atol = 1e-12
    @test norm(D) ≈ 0 atol = 1e-12
end

@testset "Test initial quantum state matches up" begin
    n = rand(1:10)
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1 + 0im
    @info("N = $n for the test")

    A = reshape(zero_state, ntuple(_ -> 2, n))

    B = reshapeTensorForMPS(A)

    C = createTensorTrainFromReshapedArray(B)

    qc = QuantumCircuit(n)

    D = getStatevectorFromTensorTrain(qc.state)

    @test D ≈ A atol = 1e-12
    @test D[1] ≈ 1.0 atol = 1e-12
    @test sum(abs2, D) ≈ 1.0 atol = 1e-12
end
