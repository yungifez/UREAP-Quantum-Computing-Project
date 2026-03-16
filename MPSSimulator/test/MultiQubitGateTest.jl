using Test, LinearAlgebra, Random

using ..MPSSimulator

using Test

@testset "Local Cx gate test on first qubit" begin
    n = 3
    N = 2^n

    psi_0 = zeros(ComplexF64, N)
    psi_0[1] = 1.0

    I2 = [1 0; 0 1]
    X = [0 1; 1 0]
    P0 = [1 0; 0 0]
    P1 = [0 0; 0 1]

    X_total = kron(I2, I2, X)

    CX_01 = kron(I2, P0) + kron(X, P1)
    CX_total = kron(I2, CX_01)

    expected_statevector = CX_total * X_total * psi_0

    qc = QuantumCircuit(n)
    x!(qc, 1)
    cx!(qc, 1, 2)

    D = getStatevectorFromTensorTrain(qc.state)

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
end

@testset "Local Cx gate test on second qubit" begin
    n = 3
    N = 2^n

    psi_0 = zeros(ComplexF64, N)
    psi_0[1] = 1.0

    I2 = [1 0; 0 1]
    X = [0 1; 1 0]
    P0 = [1 0; 0 0]
    P1 = [0 0; 0 1]

    X_total = kron(I2, X, I2)

    CX_12 = kron(P0, I2) + kron(X, P1)
    CX_total = kron(CX_12, I2)

    expected_statevector = CX_total * X_total * psi_0

    qc = QuantumCircuit(n)
    x!(qc, 2)
    cx!(qc, 2, 3)

    D = getStatevectorFromTensorTrain(qc.state)

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
end

@testset "Local Cx gate test on third qubit" begin
    n = 3
    N = 2^n

    psi_0 = zeros(ComplexF64, N)
    psi_0[1] = 1.0

    I2 = [1 0; 0 1]
    X = [0 1; 1 0]
    P0 = [1 0; 0 0]
    P1 = [0 0; 0 1]

    X_total = kron(X, I2, I2)

    CX_20 = kron(P0, I2) + kron(X, P1)
    CX_total = kron(I2, CX_20)

    expected_statevector = CX_total * X_total * psi_0

    qc = QuantumCircuit(n)
    x!(qc, 3)
    cx!(qc, 3, 1)

    D = getStatevectorFromTensorTrain(qc.state)

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
end
