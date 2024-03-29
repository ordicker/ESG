using MLUtils
using MLDatasets
using OneHotArrays: onecold

# vision datasets
for dataset = (:MNIST, :CIFAR10, :CIFAR100, :FashionMNIST)
    @eval function ($(Symbol("make_$dataset")))(; dataset_size=1000, sequence_length=50, batchsize=1024)
        ## Create the spirals
        data = $dataset()[:] # x,y 
        ## Get the labels
        ## Split the dataset
        (x_train, y_train), (x_val, y_val) = splitobs(data; at=0.8, shuffle=true)
        ## Create DataLoaders
        if data[1]|>size|>length != 4
            return (
                ## Use DataLoader to automatically minibatch and shuffle the data
                MLUtils.DataLoader(collect.((unsqueeze(x_train,dims=3), y_train)); batchsize, shuffle=true),
                ## Don't shuffle the validation data
                MLUtils.DataLoader(collect.((unsqueeze(x_val, dims=3), y_val)); batchsize, shuffle=false))
        else
            return (
                ## Use DataLoader to automatically minibatch and shuffle the data
                MLUtils.DataLoader((x_train, y_train); batchsize, shuffle=true),
                ## Don't shuffle the validation data
                MLUtils.DataLoader((x_val, y_val); batchsize, shuffle=false))
        end
    end
end

function make_poly(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 1280)), (1, 1280))
    y = evalpoly.(x, ((10, -2, 1, 17, -70),)) .+ randn(rng, (1, 1280)) .* 0.1f0
    (x_train, y_train), (x_val, y_val) = splitobs((x, y); at=0.8, shuffle=true)
    return (MLUtils.DataLoader((x_train, y_train)), MLUtils.DataLoader((x_val, y_val)))
end

# Accuracy
function accuracy(ŷ, y, topk=(1,))
    maxk = maximum(topk)

    pred_labels = partialsortperm.(eachcol(ŷ), (1:maxk,), rev=true)
    true_labels = y

    accuracies = Vector{Float32}(undef, length(topk))

    for (i, k) in enumerate(topk)
        accuracies[i] = sum(map((a, b) -> sum(view(a, 1:k) .== b),
            pred_labels,
            true_labels))
    end

    return accuracies .* 100 ./ size(y, ndims(y))
end

function simple_accuracy(ŷ, y)    
    iscorrect = map(x->x[1],argmax(ŷ,dims=1))'.==(y.+1)
    acc = round(100 * mean(iscorrect); digits=2)
end

abs2n(x::T) where {T} = abs(x)
abs2n(::Nothing) = 0.0
