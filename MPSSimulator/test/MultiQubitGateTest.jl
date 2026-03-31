using Test, LinearAlgebra, Random

include("../src/MPSSimulator.jl")
using .MPSSimulator

using Test

using Test

# Helper constants for all tests
const I2 = [1.0 0.0; 0.0 1.0]
const X = [0.0 1.0; 1.0 0.0]
const P0 = [1.0 0.0; 0.0 0.0]
const P1 = [0.0 0.0; 0.0 1.0]

@testset "Local CX Gate Tests" begin
    n = 3
    N = 2^n
    psi_init = zeros(ComplexF64, N)
    psi_init[1] = 1.0

    @testset "CX on first qubit" begin
        # X on qubit 1 (rightmost), CX control 1 target 2
        X_total = kron(I2, I2, X)
        CX_total = kron(I2, (kron(X, P1) + kron(I2, P0)))

        expected_state = CX_total * X_total * psi_init

        qc = QuantumCircuit(n)
        x!(qc, 1)
        cx!(qc, 1, 2)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX on second qubit" begin
        # X on qubit 2 (middle), CX control 2 target 3
        X_total = kron(I2, X, I2)
        # Control on 2 (middle), Target on 3 (leftmost)
        CX_total = kron(X, P1, I2) + kron(I2, P0, I2)

        expected_state = CX_total * X_total * psi_init

        qc = QuantumCircuit(n)
        x!(qc, 2)
        cx!(qc, 2, 3)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end
end

# @testset "Local Cx gate test on third qubit" begin
#     n = 3
#     N = 2^n
#
#     psi_0 = zeros(ComplexF64, N)
#     psi_0[1] = 1.0
#
#     I2 = [1 0; 0 1]
#     X = [0 1; 1 0]
#     P0 = [1 0; 0 0]
#     P1 = [0 0; 0 1]
#
#     X_total = kron(X, I2, I2)
#
#     CX_20 = kron(P0, I2) + kron(X, P1)
#     CX_total = kron(I2, CX_20)
#
#     expected_statevector = CX_total * X_total * psi_0
#
#     qc = QuantumCircuit(n)
#     x!(qc, 3)
#     cx!(qc, 3, 1)
#
#     D = getStatevectorFromTensorTrain(qc.state)
#
#     @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
# end
