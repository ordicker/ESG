using Lux, LuxCore, Optimisers, Zygote, Random, Statistics
using TensorBoardLogger, Logging
include("utils.jl") #make_<dataset>
include("ESGbackend.jl") # TODO: package
using OneHotArrays: onehotbatch
using NNlib: softmax
"""
docs
"""
function experiment()
    #### parameters
    name = "mnist_test"
    epochs = 10
    model = Chain(FlattenLayer(), Dense(784, 256, relu), Dense(256, 10), softmax)
    opt = Adam(0.01f0)#Descent(0.001f0)
    
    rng = MersenneTwister() # rng setup
    Random.seed!(rng, 12345)  
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
        mse_loss = mean(abs2, y_pred .- onehotbatch(y,0:9)) # one_hot
        return mse_loss, st, ()
    end
    #### logger setup
    # Tensorboard setup
    logger = TBLogger("logs/"*name, min_level=Logging.Info) 
    
    #### gradient method
    vjp_rule = Lux.Training.AutoZygote()
    #vjp_rule = ESG()
    
    # model and gradient warnup
    x,y = first(train_set)
    Lux.Training.compute_gradients(
        vjp_rule, loss_function, (x, y), tstate)
    
    
    with_logger(logger) do
        for epoch in 1:epochs
            println("Epoch #$epoch")
            for (x,y) in train_set 
            # Training loop
                grads, loss, stats, tstate =
                    Lux.Training.compute_gradients(
                        vjp_rule, loss_function, (x, y), tstate)
                @info "loss[train]" loss=loss 
                tstate = Lux.Training.apply_gradients(tstate, grads)
            end
            st_ = LuxCore.testmode(st)
            loss = 0.0
            for (x,y) in test_set
                # validate model
                _loss, _, _ =loss_function(model, ps, st_, (x,y))
                loss += _loss 
            end
            @info "loss[test]" loss=loss/length(test_set)
        end
    end

end
