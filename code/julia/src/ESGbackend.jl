using Lux: Training.TrainState, Training.AbstractVJP
import Lux.Training.compute_gradients
using Functors: fmap
using CUDA

struct ESG <: AbstractVJP end

backend(::ESG) = :TODO_PACKAGE

function pert_gpu(x) # TODO: config distributions
    T = eltype(x)
    e = eps(T)
    return (CUDA.rand(Bool,size(x)).-T(0.5))*T(2)*e
end

function compute_gradients(vjp::ESG, objective_function::Function, data, ts::TrainState)
    θ = ts.parameters
    ε = fmap(x->eps(eltype(x)),θ)
    dc,_,_ = loss_function(ts.model, θ, ts.states, data) # compute dc
    # generate random perturbation
    gs = fmap(x->zero(x),θ) # init zeros array
    # run perturbation N times
    N=10#length(θ)
    for _ = 1:N
        δ = fmap(pert_gpu,θ) 
        loss,_,_ = loss_function(ts.model, fmap(.+,θ,δ), ts.states, data)
        corr=fmap(x->x.*(loss-dc),δ)
        gs = fmap(.+,gs,corr)
    end
    Nε2 = fmap(x->N*x.*x,ε)
    return (fmap(./,gs,Nε2), dc, (),ts)
end

export ESG
