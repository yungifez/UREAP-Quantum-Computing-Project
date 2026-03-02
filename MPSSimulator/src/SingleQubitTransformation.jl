using LinearAlgebra

function h!(qc::QuantumCircuit, index::Int)
    M = [1 1; 1 -1] / sqrt(2)
    applySingleQubitTransformation!(qc, index, M)
end

function x!(qc::QuantumCircuit, index::Int)
    M = [0 1; 1 0]
    applySingleQubitTransformation!(qc, index, M)
end

function y!(qc::QuantumCircuit, index::Int)
    M = ComplexF64[0 -im; im 0]
    applySingleQubitTransformation!(qc, index, M)
end

function z!(qc::QuantumCircuit, index::Int)
    M = ComplexF64[1 0; 0 -1]
    applySingleQubitTransformation!(qc, index, M)
end

function s!(qc::QuantumCircuit, index::Int)
    M = ComplexF64[1 0; 0 im]
    applySingleQubitTransformation!(qc, index, M)
end

function t!(qc::QuantumCircuit, index::Int)
    M = ComplexF64[1 0; 0 exp(im * π / 4)]
    applySingleQubitTransformation!(qc, index, M)
end

function rx!(qc::QuantumCircuit, index::Int, θ::Real)
    M = ComplexF64[cos(θ / 2) -im*sin(θ / 2); -im*sin(θ / 2) cos(θ / 2)]
    applySingleQubitTransformation!(qc, index, M)
end

function ry!(qc::QuantumCircuit, index::Int, θ::Real)
    M = ComplexF64[cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)]
    applySingleQubitTransformation!(qc, index, M)
end

function rz!(qc::QuantumCircuit, index::Int, θ::Real)
    M = ComplexF64[exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]
    applySingleQubitTransformation!(qc, index, M)
end

function applySingleQubitTransformation!(qc::QuantumCircuit, index::Int, transformation::Matrix)
    siteRowSize, siteColSize = size(qc.state[index])
    transformationRowSize, transformationColSize = size(transformation)

    multiplicationFactor = Int(siteRowSize / transformationRowSize)

    gateMatrix = transformation

    # We need to inflate matrix if matrix is greater than that of the transformation
    if (multiplicationFactor > 1)
        identities = Matrix{ComplexF64}(I, multiplicationFactor, multiplicationFactor)
        gateMatrix = kron(transformation, identities)
    end

    # Order matters here T(x) = Ax
    qc.state[index] = gateMatrix * qc.state[index]
end
