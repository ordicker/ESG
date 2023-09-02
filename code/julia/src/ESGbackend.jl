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

function pert!(ε, σ)
    rand!(ε)
    ε.-=eltype(ε)(0.5)
    ε.*=σ
    return ε
end

function compute_gradients(vjp::ESG, obj_func::Function, data, ts::TrainState)
    θ = ts.parameters |> ComponentArray
    ε = zero(θ)
    dc,_,_ = obj_func(ts.model, θ, ts.states, data) # compute dc
    # generate random perturbation
    gs = zero(θ)
    # run perturbation N times
    for _ = 1:vjp.N
        pert!(ε, vjp.σ)
        loss₊,_,_ = obj_func(ts.model, θ.+ε, ts.states, data)
        loss₋,_,_ = obj_func(ts.model, θ.-ε, ts.states, data)
        gs .+= ε.*(loss₊-loss₋)
    end
    return (12*gs./vjp.N/vjp.σ^2, dc, (),ts)
end
export ESG
