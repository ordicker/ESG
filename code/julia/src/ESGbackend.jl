using Lux: Training.TrainState, Training.AbstractVJP
import Lux.Training.compute_gradients
struct PDH_VJP <: AbstractVJP end

backend(::PDH_VJP) = :TODO_PACKAGE

pert(θ) = rand(Set((1,-1).*eps(eltype(θ))),size(θ))

function compute_gradients(vjp::PDH_VJP, objective_function::Function, data, ts::TrainState)
    θ = ts.parameters |> ComponentArray
    ε = eps(eltype(θ))
    dc,_,_ = loss_function(ts.model, θ, ts.states, data) # compute dc
    # generate random perturbation
    gs = θ*0 # init zeros array
    # run perturbation N times
    N=10#length(θ)
    for s = 1:N
        δ = pert(θ)
        loss,_,_ = loss_function(ts.model, θ+δ, ts.states, data)
        gs+=(loss-dc).*δ
    end

    return (gs/N/ε^2, dc, (),ts)
end

export PDH_VJP
