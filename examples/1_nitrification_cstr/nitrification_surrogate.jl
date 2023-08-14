# Copyright (c) 2021: Matthew Wilhelm, Chenyu Wang, Matthew Stuber, 
# and the University of Connecticut (UConn).
# This code is licensed under the MIT license (see LICENSE.md for full details).
#################################################################################
# RobustHybridModels
# Examples from the paper Semi-Infinite Optimization with Hybrid Models
# https://github.com/PSORLab/RobustHybridModels
#################################################################################
# examples/1_nitrification_cstr/nitrification_surrogate.jl
# The packages required to use this file separately are: CSV, DataFrames, EAGO
# This file contains the function Ammonia_oxidation_ANN, which calculates r_AO
# based on the weights and biases of the neural network.
# Include this file by adding the line include("nitrification_surrogate.jl") in
# the main working file that calls the function Ammonia_oxidation_ANN.
# In this repository, that is examples/1_nitrification_cstr/nitrification_sip.jl.
#################################################################################

# Use the weights and biases from the given CSV
path = joinpath(@__DIR__, "nitrification_ANN_layers.csv")
nn_data = CSV.read(path, DataFrame)
size_df = size(nn_data)

# Neural network specifications
input_len = 2
hidden_len = 8
output_len = 1
depth = 2

W = Matrix{Float64}[]
b = Vector{Float64}[]

# Extract information for the first hidden layer
H1frame = filter(row -> row.Layer == "H1", nn_data)
H1frame_W = H1frame[:, [Symbol("W"*string(i)) for i = 1:input_len]]
H1frame_b = H1frame[:, [:b]]
Welement = Matrix{Float64}(H1frame_W)
belement = Matrix{Float64}(H1frame_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

# Extract information for the second to last hidden layer(s)
for i = 2:depth
    Hiframe = filter(row -> row.Layer == "H$i", nn_data)
    Hiframe_W = Hiframe[:, [Symbol("W"*string(i)) for i = 1:hidden_len]]
    Hiframe_b = Hiframe[:, [:b]]
    Welement2 = Matrix{Float64}(Hiframe_W)
    belement2 = Matrix{Float64}(Hiframe_b)
    push!(W, Welement2)
    push!(b, Float64[belement2[i] for i = 1:size(belement2,1)])
end

# Extract information for the output layer
Hoframe = filter(row -> row.Layer == "Output", nn_data)
Hoframe_W = Hoframe[:, [Symbol("W"*string(i)) for i = 1:hidden_len]]
Hoframe_b = Hoframe[:, [:b]]
Welement = Matrix{Float64}(Hoframe_W)
belement = Matrix{Float64}(Hoframe_b)
push!(W, Welement)
push!(b, Float64[belement[i] for i = 1:size(belement,1)])

# ANN for the ammonia oxidation rate, r_AO
# C_NH = x[1]
# C_O = x[2]
function Ammonia_oxidation_ANN(x)

    # Scale the input layer
    xmin = [0.0; 0.0]
    xmax = [40.0; 9.1]
    xs = (x .- xmin)./(xmax .- xmin)

    # Hidden layer 1
    yi1 = W[1]*xs .+ b[1]
    y1 = tanh.(yi1)

    # Hidden layer 2
    yi2 = W[2]*y1 .+ b[2]
    y2 = tanh.(yi2)

    # Output layer
    yi3 = W[3]*y2 .+ b[3]
    y3 = sigmoid.(yi3)

    # Return an unscaled output
    ymin = 0.0
    ymax = 0.638966464124928
    return y3[1]*(ymax - ymin) + ymin
end