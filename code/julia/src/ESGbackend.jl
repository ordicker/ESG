using Lux: Training.TrainState
using ADTypes: AbstractADType
import Lux.Training.compute_gradients
#using Functors: fmap
using ComponentArrays

struct ESG{T} <: AbstractADType
    N::Integer
    σ::T
end
ESG() = ESG(10,1f-5)

backend(::ESG) = :TODO_PACKAGE

function compute_gradients(vjp::ESG, obj_func::Function, data, ts::TrainState)
    θ = ts.parameters |> ComponentArray
    ε = zero(θ)
    dc,_,_ = obj_func(ts.model, θ, ts.states, data) # compute dc
    # generate random perturbation
    gs = zero(θ)
    tmp = zero(θ)
    rand!(tmp)
    # run perturbation N times
    for _ = 1:vjp.N
        randn!(ε)
        ε.*=vjp.σ*(tmp.<0.1)
        loss,_,_ = obj_func(ts.model, θ.+ε, ts.states, data)
        gs .+= ε.*(loss-dc)
    end
    return (gs./(vjp.N*vjp.σ^2), dc, (),ts)
end

export ESG
