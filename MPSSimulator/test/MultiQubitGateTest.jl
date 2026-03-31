using Test, LinearAlgebra, Random

using .MPSSimulator

using Test

const H = [1.0 1.0; 1.0 -1.0] / sqrt(2.0)
const I2 = [1.0 0.0; 0.0 1.0]
const X = [0.0 1.0; 1.0 0.0]
const P0 = [1.0 0.0; 0.0 0.0]
const P1 = [0.0 0.0; 0.0 1.0]

@testset "Local CX Gate Tests" begin

    n = 3
    N = 2^n
    psi_init = zeros(ComplexF64, N)
    psi_init[1] = 1.0

    @testset "CX: Control 1, Target 2" begin
        # Control: Q1 (Right), Target: Q2 (Middle), Idle: Q3 (Left)
        # Sequence: kron(Q3_Idle, Q2_Target, Q1_Control)
        X_total = kron(I2, I2, X)
        CX_total = kron(I2, X, P1) + kron(I2, I2, P0)

        expected_state = CX_total * X_total * psi_init

        qc = QuantumCircuit(n)
        x!(qc, 1)
        cx!(qc, 1, 2)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX: Control 2, Target 3" begin
        # Control: Q2 (Middle), Target: Q3 (Left), Idle: Q1 (Right)
        # Sequence: kron(Q3_Target, Q2_Control, Q1_Idle)
        X_total = kron(I2, X, I2)
        CX_total = kron(X, P1, I2) + kron(I2, P0, I2)

        expected_state = CX_total * X_total * psi_init

        qc = QuantumCircuit(n)
        x!(qc, 2)
        cx!(qc, 2, 3)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX: Control 2, Target 1" begin
        # Control: Q2 (Middle), Target: Q1 (Right), Idle: Q3 (Left)
        # Sequence: kron(Q3_Idle, Q2_Control, Q1_Target)
        X_total = kron(I2, X, I2)
        CX_total = kron(I2, P1, X) + kron(I2, P0, I2)

        expected_state = CX_total * X_total * psi_init

        qc = QuantumCircuit(n)
        x!(qc, 2)
        cx!(qc, 2, 1)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX: Control 3, Target 2" begin
        # Control: Q3 (Left), Target: Q2 (Middle), Idle: Q1 (Right)
        # Sequence: kron(Q3_Control, Q2_Target, Q1_Idle)
        X_total = kron(X, I2, I2)
        CX_total = kron(P1, X, I2) + kron(P0, I2, I2)

        expected_state = CX_total * X_total * psi_init

        qc = QuantumCircuit(n)
        x!(qc, 3)
        cx!(qc, 3, 2)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end
end

@testset "Local CX Gate Tests (7 Qubits)" begin
    n = 7
    N = 2^n
    psi_init = zeros(ComplexF64, N)
    psi_init[1] = 1.0

    @testset "CX: Control 1, Target 2" begin
        # H on Q1, CX(1, 2). Q3-Q7 are Idle.
        # Padding: I(2^(7-2)) ⊗ [Gate_on_1_and_2]
        H_total = kron(I(2^(n - 1)), H)
        CX_total = kron(I(2^(n - 2)), (kron(X, P1) + kron(I2, P0)))

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 1)
        cx!(qc, 1, 2)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX: Control 4, Target 5" begin
        # H on Q4, CX(4, 5). Q1-Q3 and Q6-Q7 are Idle.
        H_total = kron(I(2^(n - 4)), H, I(2^(4 - 1)))

        # CX block (5, 4) padded by I(2^(7-5)) on left and I(2^(4-1)) on right
        CX_core = kron(X, P1) + kron(I2, P0)
        CX_total = kron(I(2^(n - 5)), CX_core, I(2^(4 - 1)))

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 4)
        cx!(qc, 4, 5)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX: Control 7, Target 6 (Reverse)" begin
        # H on Q7 (Leftmost), CX(7, 6). Q1-Q5 are Idle.
        H_total = kron(H, I(2^(n - 1)))

        # CX block (7, 6) where high (7) controls low (6)
        CX_core = kron(P1, X) + kron(P0, I2)
        CX_total = kron(CX_core, I(2^(6 - 1)))

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 7)
        cx!(qc, 7, 6)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "CX: Control 2, Target 1 (Reverse)" begin
        # H on Q2, CX(2, 1). Q3-Q7 are Idle.
        H_total = kron(I(2^(n - 2)), H, I(2^(1)))

        # CX block (2, 1) where high (2) controls low (1)
        CX_core = kron(P1, X) + kron(P0, I2)
        CX_total = kron(I(2^(n - 2)), CX_core)

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 2)
        cx!(qc, 2, 1)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end
end

@testset "Entanglement (Bell State) Tests" begin
    n = 3
    N = 2^n
    psi_init = zeros(ComplexF64, N)
    psi_init[1] = 1.0

    @testset "Entangle: Control 1, Target 2" begin
        # H on Q1 (Right), CX(1, 2)
        # kron(Q3_Idle, Q2_Target, Q1_Control)
        H_total = kron(I2, I2, H)
        CX_total = kron(I2, X, P1) + kron(I2, I2, P0)

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 1)
        cx!(qc, 1, 2)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
        # Verify it's the state 1/sqrt(2) * (|000> + |011>)
    end

    @testset "Entangle: Control 2, Target 3" begin
        # H on Q2 (Middle), CX(2, 3)
        # kron(Q3_Target, Q2_Control, Q1_Idle)
        H_total = kron(I2, H, I2)
        CX_total = kron(X, P1, I2) + kron(I2, P0, I2)

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 2)
        cx!(qc, 2, 3)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "Entangle: Control 2, Target 1 (Reverse)" begin
        # H on Q2 (Middle), CX(2, 1)
        # kron(Q3_Idle, Q2_Control, Q1_Target)
        H_total = kron(I2, H, I2)
        CX_total = kron(I2, P1, X) + kron(I2, P0, I2)

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 2)
        cx!(qc, 2, 1)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end

    @testset "Entangle: Control 3, Target 2 (Reverse)" begin
        # H on Q3 (Left), CX(3, 2)
        # kron(Q3_Control, Q2_Target, Q1_Idle)
        H_total = kron(H, I2, I2)
        CX_total = kron(P1, X, I2) + kron(P0, I2, I2)

        expected_state = CX_total * H_total * psi_init

        qc = QuantumCircuit(n)
        h!(qc, 3)
        cx!(qc, 3, 2)

        D = getStatevectorFromTensorTrain(qc.state)
        @test vec(D) ≈ vec(expected_state) atol = 1e-12
    end
end
