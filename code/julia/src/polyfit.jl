using Lux, LuxCore, Optimisers, Zygote, Random, Statistics
using TensorBoardLogger, Logging
includet("utils.jl") #make_<dataset>
include("ESGbackend.jl") 
using OneHotArrays: onehotbatch
using NNlib: softmax
#using LuxCUDA


"""
docs
"""
function polyfit_experiment(;BP=false)
    rng = MersenneTwister() # rng setup
    Random.seed!(rng, 12345)
    #### Dataset
    train_set, test_set = make_poly(rng) #polyfit 

    #### parameters
    if BP
        name = "polyfit_BP"
        vjp_rule = Lux.Training.AutoZygote() # Backpropagation
    else
        name = "polyfit_ESG"
        vjp_rule = ESG(100,1f-7)
    end

    epochs = 200
    model = Chain(Dense(1 => 16, relu), Dense(16 => 1))
    
    opt = Adam(0.1f0)
    #### end parameters
  
    #### model setup
    dev = cpu_device()

    ps, st = Lux.setup(rng, model).|>dev
    tstate = Lux.Training.TrainState(rng, model, opt)
    #### Loss function
    function loss_function(model, ps, st, data)
        y_pred, st = Lux.apply(model, data[1], ps, st)
        mse_loss = mean(abs2, y_pred .- data[2])
        
        return mse_loss, st, ()
    end
    #### logger setup
    # Tensorboard setup
    logger = TBLogger("logs/"*name, min_level=Logging.Info) 
    
    
    # model and gradient warnup
    x,y = first(train_set)
    x,y = (x,y)|>dev
    Lux.Training.compute_gradients(
        vjp_rule, loss_function, (x, y), tstate)

    tx, ty = first(test_set)
    tx, ty = (tx, ty)|>dev
    with_logger(logger) do
        for epoch in 1:epochs
            println("Epoch #$epoch")
            # Training loop
            x,y = (x,y)|>dev
            grads, loss, st, tstate =
                Lux.Training.compute_gradients(
                    vjp_rule, loss_function, (x, y), tstate)
            @info "loss[train]" loss=loss

            @show loss
            tstate = Lux.Training.apply_gradients(tstate, grads)
            ps_ = LuxCore.testmode(tstate.parameters)
            loss = 0.0

            # validate model
            x,y = (x,y)|>dev
            _loss, _, _ = loss_function(model, tstate.parameters, tstate.states, (x,y))
 

            @info "loss[test]" loss=_loss
        end
    end
end
