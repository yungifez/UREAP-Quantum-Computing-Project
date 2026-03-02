using Test

include("../src/MPSSimulator.jl")
using .MPSSimulator

include("MPSStatePreparationTest.jl")
include("SingleQubitGateTest.jl")
include("MultiQubitGateTest.jl")
