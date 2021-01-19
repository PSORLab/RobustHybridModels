## ANN
path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Good ANNs\\Polymer\\GELU_3l_6n_1sig.csv"
nn_data_raw = CSV.read(path, DataFrame)
size_df = size(nn_data_raw)

# NEED TO ADJUST THESE FOR EACH NETWORK... TODO: DETECT FROM CSV FILE AUTOMATICALLY
# Hidden layers length (neurons per layer) varies with model, as does depth, and
# the activation function g
input_len = 3
hidden_len = 6
output_len = 1
depth = 3
g = gelu # swish1, gelu, tanh

# extracts vector of W and b that
W = Matrix{Float64}[]
b = Vector{Float64}[]

# extract first hidden layer info
H1frame = filter(row -> row.Layer === "H1", nn_data_raw)
H1frame_W = H1frame[:, [Symbol("W" * string(i)) for i = 1:input_len]]
H1frame_b = H1frame[:, [:b]]
Welement = Matrix{Float64}(H1frame_W)
belement = Matrix{Float64}(H1frame_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement, 1)])

# extract info for 2 to last hidden layer
for i = 2:depth
    Hiframe = filter(row -> row.Layer === "H$i", nn_data_raw)
    Hiframe_W = Hiframe[:, [Symbol("W" * string(i)) for i = 1:hidden_len]]
    Hiframe_b = Hiframe[:, [:b]]
    Welement2 = Matrix{Float64}(Hiframe_W)
    belement2 = Matrix{Float64}(Hiframe_b)
    push!(W, Welement2)
    push!(b, Float64[belement2[i] for i = 1:size(belement2, 1)])
end

# extract info for output layer
Hoframe = filter(row -> row.Layer === "Output", nn_data_raw)
Hoframe_W = Hoframe[:, [Symbol("W" * string(i)) for i = 1:hidden_len]]
Hoframe_b = Hoframe[:, [:b]]
Welement = Matrix{Float64}(Hoframe_W)
belement = Matrix{Float64}(Hoframe_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement, 1)])


# used to revert minmax scaling
xmin = [0.0; 0.0; 303.0]
xmax = [6.05; 0.15; 400.0]
ymax = 0.0
ymin = 1.92334e-05

function ANN1(x)
    xs = (x .- xmin)./(xmax .- xmin)

    yi1 = W[1]*xs .+ b[1]
    y1 = gelu.(yi1)

    yi2 = W[2]*y1 .+ b[2]
    y2 = gelu.(yi2)

    yi3 = W[3]*y2 .+ b[3]
    y3 = gelu.(yi3)

    yi4 = W[4]*y3 .+ b[4]
    y4 = sigmoid.(yi4)

    return y4[1]*(ymax - ymin) + ymin
end
