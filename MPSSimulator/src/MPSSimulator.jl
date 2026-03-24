module MPSSimulator

include("MPS.jl")
include("SingleQubitTransformation.jl")
include("MultiQubitTransformation.jl")

export
    # Core Structures
    QuantumCircuit,

    # Tensor Train / MPS Utilities
    reshapeTensorForMPS,
    createTensorTrainFromReshapedArray,
    getStatevectorFromTensorTrain,
    applySingleQubitTransformation!,

    # Single Qubit Gates
    h!,
    x!,
    y!,
    z!,
    s!,
    t!,
    rx!,
    ry!,
    rz!,

    # Single Qubit Gates
    cx!
end # module MpsSimulator
