using Test, LinearAlgebra, Random

using ..MPSSimulator

@testset "Test Hadamard gate works on first Qubit" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing Hadamard on Site 1 (LSB) for n = $n qubits")

    qc = QuantumCircuit(n)
    h!(qc, 1)

    D = getStatevectorFromTensorTrain(qc.state)

    H = [1 1; 1 -1] / sqrt(2)
    H_total = kron(I(2^(n - 1)), H)

    expected_statevector = H_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Test Hadamard gate works on second Qubit" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing Hadamard on Site 2 for n = $n qubits")

    qc = QuantumCircuit(n)
    h!(qc, 2)

    D = getStatevectorFromTensorTrain(qc.state)

    H = [1 1; 1 -1] / sqrt(2)
    H_total = kron(I(2^(n - 2)), H, I(2^(2 - 1)))

    expected_statevector = H_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Test Hadamard gate works on third Qubit" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing Hadamard on Site 3 (MSB) for n = $n qubits")

    qc = QuantumCircuit(n)
    h!(qc, 3)

    D = getStatevectorFromTensorTrain(qc.state)

    H = [1 1; 1 -1] / sqrt(2)
    H_total = kron(H, I(2^(3 - 1)))

    expected_statevector = H_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Test Full Superposition (H on all 3 Qubits)" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing H on all sites for n = $n (LSB=q1)")

    qc = QuantumCircuit(n)
    for i in 1:n
        h!(qc, i)
    end

    D = getStatevectorFromTensorTrain(qc.state)

    # In LSB convention, this is H ⊗ H ⊗ H
    H = [1 1; 1 -1] / sqrt(2)
    H_total = kron(H, H, H)

    expected_statevector = H_total * zero_state

    # Every element should be 1/sqrt(8) ≈ 0.3535
    @test all(x -> abs(x) ≈ 1 / sqrt(8), D)
    @test vec(D) ≈ expected_statevector atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Randomized N-Qubit Gate Test (LSB=q1)" begin
    n = rand(3:10)
    target_site = rand(1:n)
    N = 2^n
    @info("Testing random system: n = $n, target_qubit = $target_site")

    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im

    qc = QuantumCircuit(n)
    h!(qc, target_site)

    D = getStatevectorFromTensorTrain(qc.state)

    H = [1 1; 1 -1] / sqrt(2)

    I_before = I(2^(n - target_site))
    I_after = I(2^(target_site - 1))

    H_total = kron(I_before, H, I_after)
    expected_statevector = H_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Test Pauli-X gate works on first Qubit" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing Pauli-X on Site 1 (LSB) for n = $n qubits")

    qc = QuantumCircuit(n)
    x!(qc, 1)

    D = getStatevectorFromTensorTrain(qc.state)

    X = [0.0 1.0; 1.0 0.0]
    X_total = kron(I(2^(n - 1)), X)

    expected_statevector = X_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Test Pauli-X gate works on second Qubit" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing Pauli-X on Site 2 for n = $n qubits")

    qc = QuantumCircuit(n)
    x!(qc, 2)

    D = getStatevectorFromTensorTrain(qc.state)

    X = [0.0 1.0; 1.0 0.0]
    X_total = kron(I(2^(n - 2)), X, I(2^(2 - 1)))

    expected_statevector = X_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Test Pauli-X gate works on third Qubit" begin
    n = 3
    N = 2^n
    zero_state = zeros(ComplexF64, N)
    zero_state[1] = 1.0 + 0.0im
    @info("Testing Pauli-X on Site 3 (MSB) for n = $n qubits")

    qc = QuantumCircuit(n)
    x!(qc, 3)

    D = getStatevectorFromTensorTrain(qc.state)

    X = [0.0 1.0; 1.0 0.0]
    X_total = kron(X, I(2^(3 - 1)))

    expected_statevector = X_total * zero_state

    @test vec(D) ≈ vec(expected_statevector) atol = 1e-12
    @test norm(D) ≈ 1.0 atol = 1e-12
end

@testset "Randomized N-Qubit Pauli Gate Tests (LSB=q1)" begin
    n = rand(3:10)
    target_site = rand(1:n)
    N = 2^n

    # Define the initial state ONCE
    psi_zero = zeros(ComplexF64, N)
    psi_zero[1] = 1.0 + 0.0im

    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    H = [1.0 1.0; 1.0 -1.0] / sqrt(2.0)

    I_before = I(2^(n - target_site))
    I_after = I(2^(target_site - 1))

    qc_x = QuantumCircuit(n)
    x!(qc_x, target_site)
    D_x = getStatevectorFromTensorTrain(qc_x.state)

    expected_x = kron(I_before, X, I_after) * psi_zero
    @test vec(D_x) ≈ vec(expected_x) atol = 1e-12

    qc_y = QuantumCircuit(n)
    y!(qc_y, target_site)
    D_y = getStatevectorFromTensorTrain(qc_y.state)

    expected_y = kron(I_before, Y, I_after) * psi_zero
    @test vec(D_y) ≈ vec(expected_y) atol = 1e-12

    qc_z = QuantumCircuit(n)
    h!(qc_z, target_site)
    z!(qc_z, target_site)
    D_z = getStatevectorFromTensorTrain(qc_z.state)

    expected_z = kron(I_before, Z * H, I_after) * psi_zero
    @test vec(D_z) ≈ vec(expected_z) atol = 1e-12
end

@testset "Sequential Rotation & Pauli Test (State Tracking)" begin
    n = rand(6:13)
    N = 2^n
    qc = QuantumCircuit(n)

    # Reference math state
    ref_state = zeros(ComplexF64, N)
    ref_state[1] = 1.0 + 0.0im

    @info "Starting rotation stress test on $n qubits..."

    for i in 1:15
        target = rand(1:n)
        θ = rand() * 2π

        choice = rand(1:6)

        local_gate_mat = ComplexF64[1 0; 0 1] # Placeholder

        if choice == 1
            x!(qc, target)
            local_gate_mat = ComplexF64[0 1; 1 0]
            @info "Step $i: X on q$target"
        elseif choice == 2
            h!(qc, target)
            local_gate_mat = ComplexF64[1 1; 1 -1] / sqrt(2)
            @info "Step $i: H on q$target"
        elseif choice == 3
            rx!(qc, target, θ)
            local_gate_mat = ComplexF64[cos(θ / 2) -im*sin(θ / 2); -im*sin(θ / 2) cos(θ / 2)]
            @info "Step $i: Rx(θ) on q$target with θ=$(round(θ, digits=3))"
        elseif choice == 4
            ry!(qc, target, θ)
            local_gate_mat = ComplexF64[cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)]
            @info "Step $i: Ry(θ) on q$target with θ=$(round(θ, digits=3))"
        elseif choice == 5
            rz!(qc, target, θ)
            local_gate_mat = ComplexF64[exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]
            @info "Step $i: Rz(θ) on q$target with θ=$(round(θ, digits=3))"
        elseif choice == 6
            y!(qc, target)
            local_gate_mat = ComplexF64[0 -im; im 0]
            @info "Step $i: Y on q$target"
        end

        I_before = I(2^(n - target))
        I_after = I(2^(target - 1))
        operator = kron(I_before, local_gate_mat, I_after)
        ref_state = operator * ref_state

        current_sim_state = getStatevectorFromTensorTrain(qc.state)

        @test vec(current_sim_state) ≈ vec(ref_state) atol = 1e-12
        @test norm(current_sim_state) ≈ 1.0 atol = 1e-12
    end
end

