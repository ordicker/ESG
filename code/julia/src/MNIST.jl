using Lux, LuxCore, Optimisers, Zygote, Random, Statistics
using TensorBoardLogger, Logging
include("utils.jl") #make_<dataset>
include("ESGbackend.jl") 
using OneHotArrays: onehotbatch
using NNlib: softmax
#using LuxCUDA
"""
docs
"""
function MNIST_experiment(;BP=false)
    #### Dataset
    train_set, test_set = make_MNIST() #MNIST

    #### parameters
    if BP
        name = "MNIST_BP"
        vjp_rule = Lux.Training.AutoZygote() # Backpropagation
    else
        name = "MNIST_ESG_nondiff"
        vjp_rule = ESG(100,1f-6)
    end

    epochs = 100
    model = Chain(FlattenLayer(), Dense(784, 256, sign), Dense(256, 10), softmax)
    
    rng = MersenneTwister() # rng setup
    Random.seed!(rng, 12345)
    opt = Adam(0.1f0)
    #### end parameters
  
    #### model setup
    dev = cpu_device()

    ps, st = Lux.setup(rng, model).|>dev
    tstate = Lux.Training.TrainState(rng, model, opt)
    #### Loss function
    function loss_function(model, ps, st, data)
        x,y = data
        y_pred, st = Lux.apply(model, x, ps, st)
        #loss = mean(abs2, y_pred .- onehotbatch(y,0:9)) # one_hot
        loss = -sum(log.(y_pred.+eps(eltype(y_pred))).*onehotbatch(y,0:9))/length(y)
        return loss, st, ()
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
            for (x,y) in train_set 
                # Training loop
                x,y = (x,y)|>dev
                grads, loss, st, tstate =
                    Lux.Training.compute_gradients(
                        vjp_rule, loss_function, (x, y), tstate)
                @info "loss[train]" loss=loss

                @show simple_accuracy(model(tx,
                                     LuxCore.testmode(tstate.parameters),
                                     tstate.states)[1], ty)
                tstate = Lux.Training.apply_gradients(tstate, grads)
            end
            ps_ = LuxCore.testmode(tstate.parameters)
            loss = 0.0
            for (x,y) in test_set
                # validate model
                x,y = (x,y)|>dev
                _loss, _, _ =loss_function(model, tstate.parameters, tstate.states, (x,y))
                loss += _loss 
            end
            @info "loss[test]" loss=loss/length(test_set)
        end
    end
end
