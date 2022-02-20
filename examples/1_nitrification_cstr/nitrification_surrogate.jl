#path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\GELU4_ANN.csv"
#path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\TANH2_ANN.csv"
path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\SILU2_ANN.csv"
nn_data_raw = CSV.read(path, DataFrame)
size_df = size(nn_data_raw)

input_len = 2
hidden_len = 6
output_len = 1
depth = 2
g = swish1


# extracts vector of W and b that
W = Matrix{Float64}[]
b = Vector{Float64}[]

# extract first hidden layer info
H1frame = filter(row -> row.Layer === "H1", nn_data_raw)
H1frame_W = H1frame[:, [Symbol("W"*string(i)) for i=1:input_len]]
H1frame_b = H1frame[:, [:b]]
Welement = Matrix{Float64}(H1frame_W)
belement = Matrix{Float64}(H1frame_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

# extract info for 2 to last hidden layer
for i = 2:depth
    Hiframe = filter(row -> row.Layer === "H$i", nn_data_raw)
    Hiframe_W = Hiframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
    Hiframe_b = Hiframe[:, [:b]]
    Welement2 = Matrix{Float64}(Hiframe_W)
    belement2 = Matrix{Float64}(Hiframe_b)
    push!(W, Welement2)
    push!(b, Float64[belement2[i] for i = 1:size(belement2,1)])
end

# extract info for output layer
Hoframe = filter(row -> row.Layer === "Output", nn_data_raw)
Hoframe_W = Hoframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
Hoframe_b = Hoframe[:, [:b]]
Welement = Matrix{Float64}(Hoframe_W)
belement = Matrix{Float64}(Hoframe_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

#mutable struct ANN1 <: Function
#end

xmin = zeros(2)
xmax = [40.0; 9.1]
ymin = 0.0
ymax = 0.638966464124928

#=
function ANN1(x)
    xs = (x .- xmin)./(xmax .- xmin)

    yi1 = W[1]*xs .+ b[1]
    y1 = swish1.(yi1)
    #@show yi1
    #@show y1

    yi2 = W[2]*y1 .+ b[2]
    y2 = swish1.(yi2)

    yi3 = W[3]*y2 .+ b[3]
    y3 = swish1.(yi3)

    yi4 = W[4]*y3 .+ b[4]
    y4 = swish1.(yi4)

    yi5 = W[5]*y4 .+ b[5]
    y5 = sigmoid.(yi5)

    y5[1]*(ymax - ymin) + ymin
end
=#

function ANN1(x)

    #@show x
    xs = (x .- xmin)./(xmax .- xmin)

    yi1 = W[1]*xs .+ b[1]
    y1 = swish1.(yi1)
    #@show yi1
    #@show y1

    yi2 = W[2]*y1 .+ b[2]
    y2 = swish1.(yi2)
    #@show yi2
    #@show y2

    yi3 = W[3]*y2 .+ b[3]
    y3 = sigmoid.(yi3)
    #@show yi4
    #@show y4

    #@show y4[1]*(ymax - ymin) + ymin
    return y3[1]*(ymax - ymin) + ymin
end

#=
# builds a nonallocating multilayer perceptron (vanilla ANN) with static storage types
calc_block = quote end

# used to revert minmax scaling
xmin = zeros(2)
xmax = [40.0; 9.1]

# unpacks input variables (CHECKED)
for i = 1:input_len
    sym = Symbol("Input_"*string(i))
    #push!(calc_block.args, :($sym = (x[$i] - $(xmin[i]))/($(xmax[i]) - $(xmin[i]))))
    push!(calc_block.args, :($sym = x[$i]))
end

# input to first hidden layer (CHECKED)
for i = 1:hidden_len
    sym = Symbol("H1_"*string(i))
    sum_expr = :($(b[1][i]))
    for j = 1:input_len
        prior_sym =  Symbol("Input_"*string(j))
        sum_expr = :($sum_expr + $(W[1][i,j])*$prior_sym)
    end
    push!(calc_block.args, :($sym = f($sum_expr)))
end

# 2 to last hidden_layers
for k = 2:depth
    @show k
    for i = 1:hidden_len
        sym = Symbol("H$(k)_"*string(i))
        sum_expr = :($(b[1][i]))
        for j = 1:input_len
            prior_sym =  Symbol("H$(k-1)_"*string(j))
            sum_expr = :($sum_expr + $(W[k][i,j])*$prior_sym)
        end
        push!(calc_block.args, :($sym = f($sum_expr)))
    end
end

output_tuple = :((Output_1,))
for i = 1:output_len
    sym = Symbol("Output_"*string(i))
    if i > 1
        push!(output_tuple, sym)
    end
    sum_expr = :($(b[end][i]))
    for j = 1:hidden_len
        prior_sym = Symbol("H$(depth)_"*string(j))
        sum_expr = :($sum_expr + $(W[end][i,j])*$prior_sym)
    end
    push!(calc_block.args, :($sym = $sum_expr))
end

@generated function ANN(x, f)
    return quote
        $calc_block
        return $output_tuple
    end
end

ANN1(x) = ANN(x, g)[1]

function ANN_GLM1(x)
    ANN1(x) + 0.0020*x[1] + 0.0245*x[2] + 0.4163
end
=#

#=
path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\GELU1_ANN.csv"
nn_data_raw = CSV.read(path, DataFrame)
size_df = size(nn_data_raw)

input_len = 2
hidden_len = 5
output_len = 1
depth = 1
g = swish1

# extracts vector of W and b that
W = Matrix{Float64}[]
b = Vector{Float64}[]

# extract first hidden layer info
H1frame = filter(row -> row.Layer === "H1", nn_data_raw)
H1frame_W = H1frame[:, [Symbol("W"*string(i)) for i=1:input_len]]
H1frame_b = H1frame[:, [:b]]
Welement = Matrix{Float64}(H1frame_W)
belement = Matrix{Float64}(H1frame_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

# extract info for 2 to last hidden layer
for i = 2:depth
    Hiframe = filter(row -> row.Layer === "H$i", nn_data_raw)
    Hiframe_W = Hiframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
    Hiframe_b = Hiframe[:, [:b]]
    Welement2 = Matrix{Float64}(Hiframe_W)
    belement2 = Matrix{Float64}(Hiframe_b)
    push!(W, Welement2)
    push!(b, Float64[belement2[i] for i = 1:size(belement2,1)])
end

# extract info for output layer
Hoframe = filter(row -> row.Layer === "Output", nn_data_raw)
Hoframe_W = Hoframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
Hoframe_b = Hoframe[:, [:b]]
Welement = Matrix{Float64}(Hoframe_W)
belement = Matrix{Float64}(Hoframe_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

# builds a nonallocating multilayer perceptron (vanilla ANN) with static storage types
calc_block = quote end


# used to revert minmax scaling
xmin = zeros(2)
xmax = [40.0; 9.1]

# unpacks input variables (CHECKED)
for i = 1:input_len
    sym = Symbol("Input_"*string(i))
    push!(calc_block.args, :($sym = (x[$i] - $(xmin[i]))/($(xmax[i]) - $(xmin[i]))))
end

# input to first hidden layer (CHECKED)
for i = 1:hidden_len
    sym = Symbol("H1_"*string(i))
    sum_expr = :($(b[1][i]))
    for j = 1:input_len
        prior_sym =  Symbol("Input_"*string(j))
        sum_expr = :($sum_expr + $(W[1][i,j])*$prior_sym)
    end
    push!(calc_block.args, :($sym = f($sum_expr)))
end

# 2 to last hidden_layers
for k = 2:depth
    @show k
    for i = 1:hidden_len
        sym = Symbol("H$(k)_"*string(i))
        sum_expr = :($(b[1][i]))
        for j = 1:input_len
            prior_sym =  Symbol("H$(k-1)_"*string(j))
            sum_expr = :($sum_expr + $(W[k][i,j])*$prior_sym)
        end
        push!(calc_block.args, :($sym = f($sum_expr)))
    end
end

output_tuple = :((Output_1,))
for i = 1:output_len
    sym = Symbol("Output_"*string(i))
    if i > 1
        push!(output_tuple, sym)
    end
    sum_expr = :($(b[end][i]))
    for j = 1:hidden_len
        prior_sym = Symbol("H$(depth)_"*string(j))
        sum_expr = :($sum_expr + $(W[end][i,j])*$prior_sym)
    end
    push!(calc_block.args, :($sym = $sum_expr))
end

@generated function ANN(x, f)
    return quote
        $calc_block
        return $output_tuple
    end
end

ANN1(x) = ANN(x, g)[1]

function ANN_GLM1(x)
    ANN1(x) + 0.0020*x[1] + 0.0245*x[2] + 0.4163
end
=#
