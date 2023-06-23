include("ESGbackend.jl") # TODO: package
# Write your package code here.
main() = println("hi!")

###############################################################################
#                                 Lux example                                 #
###############################################################################
using Lux, LuxCore, ComponentArrays
using DataLoaders
using LinearAlgebra # for svd
using NNlib, Optimisers, Random, Zygote, Statistics

function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
    return (x, y)
end

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])

    return mse_loss, st, ()
end


function lux_example()
    rng = MersenneTwister()
    Random.seed!(rng, 12345)

    (x, y) = generate_data(rng)
    
    model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

    opt = Adam(0.03f0)
    ps, st = Lux.setup(rng, model)

    #return model, ps, st
    #ESGgrads(loss_function, model, ps, st, (x, y))

    tstate = Lux.Training.TrainState(rng, model, opt)

    #vjp_rule = Lux.Training.ZygoteVJP()
    vjp_rule = ESG()

    for epoch in 1:250
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp_rule, loss_function,
                                                                    (x, y), tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    tstate
end

