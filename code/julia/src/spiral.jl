using  MLUtils, Optimisers, Random, Statistics , Zygote
using TensorBoardLogger, Logging
using Lux
include("ESGbackend.jl")

# ## Dataset

# We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise
# spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give
# us sequences of size 2 × seq_len × batch_size and we need to predict a binary value
# whether the sequence is clockwise or anticlockwise.

function get_dataloaders(; dataset_size=1000, sequence_length=50)
    ## Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    ## Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(d[1][:, (sequence_length + 1):end], :,
        sequence_length, 1) for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    ## Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    ## Create DataLoaders
    return (
        ## Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        ## Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end

# ## Creating a Classifier

# We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model
# since it will contain a lstm block and a classifier head.

# We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the
# parameters and states are automatically populated and we don't have to define
# `Lux.initialparameters` and `Lux.initialstates`.

# To understand more about container layers, please look at
# [Container Layer](http://lux.csail.mit.edu/stable/manual/interface/#container-layer).

struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end

# We won't define the model from scratch but rather use the [`Lux.LSTMCell`](@ref) and
# [`Lux.Dense`](@ref).

function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(LSTMCell(in_dims => hidden_dims),
        Dense(hidden_dims => out_dims, sigmoid))
end

# We can use default Lux blocks -- `Recurrence(LSTMCell(in_dims => hidden_dims)` -- instead
# of defining the following. But let's still do it for the sake of it.

# Now we need to define the behavior of the Classifier when it is invoked.

function (s::SpiralClassifier)(x::AbstractArray{T, 3},
    ps::NamedTuple,
    st::NamedTuple) where {T}
    ## First we will have to run the sequence through the LSTM Cell
    ## The first call to LSTM Cell will create the initial hidden state
    ## See that the parameters and states are automatically populated into a field called
    ## `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    ## and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    ## Now that we have the hidden state and memory in `carry` we will pass the input and
    ## `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    ## After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    ## Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end

# ## Defining Accuracy, Loss and Optimiser

# Now let's define the binarycrossentropy loss. Typically it is recommended to use
# `logitbinarycrossentropy` since it is more numerically stable, but for the sake of
# simplicity we will use `binarycrossentropy`.

function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function loss_function(model, ps, st, data)
    x, y = data
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), st, ()
end

matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)

# ## Training the Model

function spiral_experiment(;BP=false)    
    ## Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    #### parameters
    if BP
        name = "spiral_BP"
        vjp_rule = Lux.Training.AutoZygote() # Backpropagation
    else
        name = "spiral_ESG"
        vjp_rule = ESG(100,1f-6)
    end

    epochs = 100
    ## Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model) 

    dev = cpu_device()
    ps = ps |> dev
    st = st |> dev

    ## Create the optimiser
    #opt = create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    
    tstate = Lux.Training.TrainState(rng, model, opt, transform_variables=dev)

    #### logger setup
    # Tensorboard setup
    logger = TBLogger("logs/"*name, min_level=Logging.Info) 

    x,y = first(train_loader)
    x,y = (x,y)|>dev
    grads, loss, st, tstate =
        Lux.Training.compute_gradients(
            vjp_rule, loss_function, (x, y), tstate)

    with_logger(logger) do
        for epoch in 1:epochs
            println("Epoch #$epoch")
            ## Train the model          
            for (x, y) in train_loader
                x = x |> dev
                y = y |> dev
                grads, loss, _, tstate =
                    Lux.Training.compute_gradients(
                        vjp_rule, loss_function, (x, y), tstate)
                tstate = Lux.Training.apply_gradients(tstate, grads)

                println("Epoch [$epoch]: Loss $loss")
            end
            @info "loss[train]" loss=loss
            ## Validate the model
            tstate_ = deepcopy(tstate)
            for (x, y) in train_loader
                x = x |> dev
                y = y |> dev
                _, loss_, _, tstate_ =
                    Lux.Training.compute_gradients(
                        vjp_rule, loss_function, (x, y), tstate_)                
                @info "loss[test]" loss=loss_
            end
        end
    end
end
