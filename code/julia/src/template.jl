using Lux, LuxCore, Optimisers, Zygote, Random, Statistics
using TensorBoardLogger, Logging
include("utils.jl") #make_<dataset>
includet("ESGbackend.jl") # TODO: package
using OneHotArrays: onehotbatch
using NNlib: softmax
"""
docs
"""
function experiment()
    #### parameters
    name = "mnist_test"
    epochs = 5
    model = Chain(FlattenLayer(), Dense(784, 256, relu), Dense(256, 10), softmax)
    #model = Chain(
    #    Conv((5, 5), 1=>6, relu),
    #    MaxPool((2, 2)),
    #    Conv((5, 5), 6=>16, relu),
    #    MaxPool((2, 2)),
    #    flatten,
    #    Dense(256, 120, relu), 
    #    Dense(120, 84, relu), 
    #    Dense(84, 10),
    #    softmax
    #)
    rng = MersenneTwister() # rng setup
    Random.seed!(rng, 12345)

    opt = Adam(30f-4)
    #opt = Momentum(30f-2, 0.99)
    #opt = Descent(30f-2)
    #### gradient method
    #vjp_rule = Lux.Training.AutoZygote()
    vjp_rule = ESG(10,1f-7)

    #### end parameters

    #### Dataset
    train_set, test_set = make_MNIST() #mnist 
    
    #### model setup
    ps, st = Lux.setup(rng, model)#.|>gpu_device()
    tstate = Lux.Training.TrainState(rng, model, opt)#, transform_variables=gpu)
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
    Lux.Training.compute_gradients(
        vjp_rule, loss_function, (x, y), tstate)

    tx, ty = first(test_set)
    with_logger(logger) do
        for epoch in 1:epochs
            println("Epoch #$epoch")
            for (x,y) in train_set 
                # Training loop
                grads, loss, st, tstate =
                    Lux.Training.compute_gradients(
                        vjp_rule, loss_function, (x, y), tstate)
                @info "loss[train]" loss=loss
                @show loss
                #@show maximum(st)
                @show sum(abs2n, grads|>ComponentArray)
                @show simple_accuracy(model(tx,
                                     tstate.parameters,
                                     tstate.states)[1], ty)
                tstate = Lux.Training.apply_gradients(tstate, grads)
            end
            #ps_ = LuxCore.testmode(tstate.parameters)
            loss = 0.0
            for (x,y) in test_set
                # validate model
                _loss, _, _ =loss_function(model, tstate.parameters, tstate.states, (x,y))
                loss += _loss 
            end
            @info "loss[test]" loss=loss/length(test_set)
        end
    end

end
