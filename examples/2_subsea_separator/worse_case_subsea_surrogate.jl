path_valid = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Good ANNs\\Subsea Classifier\\SILU_2l_2n_1sig.csv"
nn_data_raw_valid = CSV.read(path_valid, DataFrame)
size_df_valid = size(nn_data_raw_valid)

input_len = 4
hidden_len = 2
output_len = 1
depth = 2
g = swish1

xmin = [8.23365; 0.35171; 0.35; 0.10012]
xmax = [19.5046; 0.8; 0.499985; 0.25]

ymin1 = 0.0
ymax1 = 1.0

# extracts vector of W and b that
W = Matrix{Float64}[]
b = Vector{Float64}[]

# extract first hidden layer info
H1frame = filter(row -> row.Layer === "H1", nn_data_raw_valid)
H1frame_W = H1frame[:, [Symbol("W"*string(i)) for i=1:input_len]]
H1frame_b = H1frame[:, [:b]]
Welement = Matrix{Float64}(H1frame_W)
belement = Matrix{Float64}(H1frame_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

# extract info for 2 to last hidden layer
for i = 2:depth
    Hiframe = filter(row -> row.Layer === "H$i", nn_data_raw_valid)
    Hiframe_W = Hiframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
    Hiframe_b = Hiframe[:, [:b]]
    Welement2 = Matrix{Float64}(Hiframe_W)
    belement2 = Matrix{Float64}(Hiframe_b)
    push!(W, Welement2)
    push!(b, Float64[belement2[i] for i = 1:size(belement2,1)])
end

# extract info for output layer
Hoframe = filter(row -> row.Layer === "Output", nn_data_raw_valid)
Hoframe_W = Hoframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
Hoframe_b = Hoframe[:, [:b]]
Welement = Matrix{Float64}(Hoframe_W)
belement = Matrix{Float64}(Hoframe_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

function ANN_valid(x)
    xs = (x .- xmin)./(xmax .- xmin)

    yi1 = W[1]*xs .+ b[1]
    y1 = swish1.(yi1)

    yi2 = W[2]*y1 .+ b[2]
    y2 = swish1.(yi2)

    yi3 = W[3]*y2 .+ b[3]
    y3 = sigmoid.(yi3)

    return y3[1]*(ymax1 - ymin1) + ymin1
    #return y4[1]*(ymax1 - ymin1) + ymin1
end


path_perform = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Good ANNs\\Subsea Surrogate\\SILU_2l_12n_4sig.csv"
nn_data_raw_perform = CSV.read(path_perform, DataFrame)
size_df_perform = size(nn_data_raw_perform)

input_len = 4
hidden_len = 12
output_len = 4
depth = 2
g = swish1

# extracts vector of W and b that
W2 = Matrix{Float64}[]
b2 = Vector{Float64}[]

# extract first hidden layer info
H1frame = filter(row -> row.Layer === "H1", nn_data_raw_perform)
H1frame_W = H1frame[:, [Symbol("W"*string(i)) for i=1:input_len]]
H1frame_b = H1frame[:, [:b]]
Welement = Matrix{Float64}(H1frame_W)
belement = Matrix{Float64}(H1frame_b)
push!(W2, Welement)
push!(b2, Float64[belement[i] for i = 1:size(belement,1)])

# extract info for 2 to last hidden layer
for i = 2:depth
    Hiframe = filter(row -> row.Layer === "H$i", nn_data_raw_perform)
    Hiframe_W = Hiframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
    Hiframe_b = Hiframe[:, [:b]]
    Welement2 = Matrix{Float64}(Hiframe_W)
    belement2 = Matrix{Float64}(Hiframe_b)
    push!(W2, Welement2)
    push!(b2, Float64[belement2[i] for i = 1:size(belement2,1)])
end

# extract info for output layer
Hoframe = filter(row -> row.Layer === "Output", nn_data_raw_perform)
Hoframe_W = Hoframe[:, [Symbol("W"*string(i)) for i=1:hidden_len]]
Hoframe_b = Hoframe[:, [:b]]
Welement = Matrix{Float64}(Hoframe_W)
belement = Matrix{Float64}(Hoframe_b)
push!(W2, Welement)
push!(b2, Float64[belement[i] for i = 1:size(belement,1)])


ymin2 = [0.462387; 0.00946356; 4e+06; 825.87]
ymax2 = [0.799078; 0.138537; 4.01e+06; 911.39]

function ANN_perform(x)
    xs = (x .- xmin)./(xmax .- xmin)

    yi1 = W2[1]*xs .+ b2[1]
    y1 = swish1.(yi1)

    yi2 = W2[2]*y1 .+ b2[2]
    y2 = swish1.(yi2)

    yi3 = W2[3]*y2 .+ b2[3]
    y3 = sigmoid.(yi3)

    ys5 = (ymax2 .- ymin2).*y3 .+ ymin2

    #return y5[1], y5[2], y5[3], y5[4]
    return ys5[1], ys5[2], ys5[3], ys5[4]
end
