using .MPSSimulator
using Test
using LinearAlgebra
using Random

function is_physically_equivalent(u, v; atol=1e-12)
    u_vec = vec(u)
    v_vec = vec(v)
    return isapprox(abs(dot(u_vec, v_vec)), 1.0, atol=atol)
end

const I2 = [1.0 0.0; 0.0 1.0]
const X = [0.0 1.0; 1.0 0.0]
const Y = [0.0 -im; im 0.0]
const Z = [1.0 0.0; 0.0 -1.0]
const H = [1.0 1.0; 1.0 -1.0] / sqrt(2.0)
const P0 = [1.0 0.0; 0.0 0.0]
const P1 = [0.0 0.0; 0.0 1.0]

@testset "3-Qubit Adjacent Gate Tests" begin
    n = 3
    N = 2^n
    init = zeros(ComplexF64, N); init[1] = 1.0
    
    # Adjacent pairs for n=3: (1,2), (2,1), (2,3), (3,2)
    adj_pairs = [(1,2), (2,1), (2,3), (3,2)]

    for (c, t) in adj_pairs
        @testset "Local CX: C$c, T$t" begin
            qc = QuantumCircuit(n)
            x!(qc, c)
            cx!(qc, c, t)
            
            term1 = kron([i == c ? P1 : i == t ? X : I2 for i in n:-1:1]...)
            term2 = kron([i == c ? P0 : I2 for i in n:-1:1]...)
            expected = (term1 + term2) * kron([i == c ? X : I2 for i in n:-1:1]...) * init
            @test is_physically_equivalent(getStatevectorFromTensorTrain(qc.state), expected)
        end

        @testset "Local CY: C$c, T$t" begin
            qc = QuantumCircuit(n)
            x!(qc, c)
            cy!(qc, c, t)
            
            term1 = kron([i == c ? P1 : i == t ? Y : I2 for i in n:-1:1]...)
            term2 = kron([i == c ? P0 : I2 for i in n:-1:1]...)
            expected = (term1 + term2) * kron([i == c ? X : I2 for i in n:-1:1]...) * init
            @test is_physically_equivalent(getStatevectorFromTensorTrain(qc.state), expected)
        end

        @testset "Local CZ: C$c, T$t" begin
            qc = QuantumCircuit(n)
            x!(qc, c); x!(qc, t)
            cz!(qc, c, t)
            
            term1 = kron([i == c ? P1 : i == t ? Z : I2 for i in n:-1:1]...)
            term2 = kron([i == c ? P0 : I2 for i in n:-1:1]...)
            prep = kron([i == c || i == t ? X : I2 for i in n:-1:1]...)
            expected = (term1 + term2) * prep * init
            @test is_physically_equivalent(getStatevectorFromTensorTrain(qc.state), expected)
        end
    end

    for q1 in 1:(n-1)
        q2 = q1 + 1
        @testset "Local SWAP: $q1, $q2" begin
            qc = QuantumCircuit(n)
            x!(qc, q1)
            swap!(qc, q1, q2)
            
            expected = kron([i == q2 ? X : I2 for i in n:-1:1]...) * init
            @test is_physically_equivalent(getStatevectorFromTensorTrain(qc.state), expected)
        end
    end
end

@testset "7-Qubit Adjacent Scaling Tests" begin
    n = 7
    N = 2^n
    init = zeros(ComplexF64, N); init[1] = 1.0

    # Testing boundaries and middle for 7 qubits
    test_adj = [(1, 2), (2, 1), (3, 4), (4, 3), (6, 7), (7, 6)]

    for (c, t) in test_adj
        @testset "CX 7Q Local: C$c, T$t" begin
            qc = QuantumCircuit(n)
            h!(qc, c)
            cx!(qc, c, t)
            
            term1 = kron([i == c ? P1 : i == t ? X : I2 for i in n:-1:1]...)
            term2 = kron([i == c ? P0 : I2 for i in n:-1:1]...)
            h_gate = kron([i == c ? H : I2 for i in n:-1:1]...)
            
            expected = (term1 + term2) * h_gate * init
            @test is_physically_equivalent(getStatevectorFromTensorTrain(qc.state), expected)
        end
    end
end
