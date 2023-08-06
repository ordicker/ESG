using MLUtils
using MLDatasets

# vision datasets
for dataset = (:MNIST, :CIFAR10, :CIFAR100, :FashionMNIST)
    @eval function ($(Symbol("make_$dataset")))(; dataset_size=1000, sequence_length=50, batchsize=128)
        ## Create the spirals
        data = $dataset()[:] # x,y 
        ## Get the labels
        ## Split the dataset
        (x_train, y_train), (x_val, y_val) = splitobs(data; at=0.8, shuffle=true)
        ## Create DataLoaders
        return (
            ## Use DataLoader to automatically minibatch and shuffle the data
            MLUtils.DataLoader(collect.((unsqueeze(x_train,dims=3), y_train)); batchsize, shuffle=true),
            ## Don't shuffle the validation data
            MLUtils.DataLoader(collect.((unsqueeze(x_val, dims=3), y_val)); batchsize, shuffle=false))
    end
end



