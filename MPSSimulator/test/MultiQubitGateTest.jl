using Test, LinearAlgebra, Random

using ..MPSSimulator

@testset "Cx gate test" begin
    n = 3
    N = 2^n

    psi_0 = zeros(ComplexF64, N)
    psi_0[1] = 1.0 + 0im

    X = [0 1; 1 0]
    I2 = [1 0; 0 1]

    CX_rev = [
        1 0 0 0;
        0 0 0 1;
        0 0 1 0;
        0 1 0 0
    ]

    X_total = kron(kron(I2, X), I2)

    CX_total = kron(CX_rev, I2)

    expected_statevector = CX_total * X_total * psi_0

    qc = QuantumCircuit(n)
    x!(qc, 1)
    cx!(qc, 1, 2)

    D = getStatevectorFromTensorTrain(qc.state)

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
end
