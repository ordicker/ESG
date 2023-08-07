includet("ESGbackend.jl") # TODO: package

###############################################################################
#                                 Lux example                                 #
###############################################################################
using Lux, LuxCore#, LuxCUDA
using DataLoaders
using NNlib, Optimisers, Random, Zygote, Statistics
using TensorBoardLogger, Logging


function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((10, -2, 1, 17, -70),)) .+ randn(rng, (1, 128)) .* 0.1f0
    return (x, y)
end

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])

    return mse_loss, st, ()
end


function polyfit()
    logger = TBLogger("logs/polyfit", min_level=Logging.Info)
    rng = MersenneTwister()
    Random.seed!(rng, 12345)

    (x, y) = generate_data(rng)|>gpu_device()
    
    model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

    opt = Adam(0.1f0)
    #opt = OptimiserChain(ClipGrad(0.01f0), Adam(1f0));
    ps, st = Lux.setup(rng, model)#.|>gpu_device()

    tstate = Lux.Training.TrainState(rng, model, opt)#, transform_variables=gpu)

    #vjp_rule = Lux.Training.AutoZygote()
    vjp_rule = ESG()
    
    with_logger(logger) do
        for epoch in 1:10000
            grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp_rule, loss_function,
                                                                    (x, y), tstate)
            @info "polyfit" epoch=epoch loss=loss max_var = maximum(stats)
            @show loss
            @show maximum(stats)
            tstate = Lux.Training.apply_gradients(tstate, grads)
        end
    end
end

