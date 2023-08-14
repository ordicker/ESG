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
    sumsq = zero(θ)
    # run perturbation N times
    for _ = 1:vjp.N
        randn!(ε)
        ε.*=vjp.σ
        loss,_,_ = obj_func(ts.model, θ.+ε, ts.states, data)
        gs .+= ε.*(loss-dc)
        sumsq .+= gs.*gs
    end
    return (gs/vjp.N/vjp.σ*1f7, dc, ((sumsq.-gs.*gs/vjp.N)/(vjp.N-1)),ts) #./(vjp.N*vjp.σ)
end
export ESG
