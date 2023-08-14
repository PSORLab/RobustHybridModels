# Copyright (c) 2021: Matthew Wilhelm, Chenyu Wang, Matthew Stuber, 
# and the University of Connecticut (UConn).
# This code is licensed under the MIT license (see LICENSE.md for full details).
#################################################################################
# RobustHybridModels
# Examples from the paper Semi-Infinite Optimization with Hybrid Models
# https://github.com/PSORLab/RobustHybridModels
#################################################################################
# examples/1_nitrification_cstr/surrogate_model/nitrification_data_generation.jl
# This file is used to generate Latin hypercube samples (LHC) of the given
# ammonia oxidation rate at specified domains. The data is then saved to a CSV
# which is used in 
# examples/1_nitrification_cstr/surrogate_model/nitrification_training.py to
# create a neural network for the ammonia oxidation rate.
#################################################################################

using LatinHypercubeSampling, Statistics, DelimitedFiles

# Kinetic model for the ammonia oxidation rate
# C_NH = x[1]
# C_O = x[2]
function Ammonia_oxidation_rate(x)
    r_AOmax = 0.67/60   # mg N-NH4+/(g VSS_AO*s)
    K_SAO = 0.24        # mg N-NH4+/L
    K_IAO = 6200.0      # mg N-NH4+/L
    K_OAO = 0.3         # mg/L
    r_AO = r_AOmax*(x[1]/(K_SAO + x[1] + x[1]^2/K_IAO))*(x[2]/(K_OAO + x[2]))   # mg N-NH4+/(g VSS_AO*s)
    return r_AO
end

# Generate LHC data using the first domain [0, 4] x [0, 1]
Data_LHC1 = randomLHC(10000, 2)
Data_input1 = scaleLHC(Data_LHC1, [(0.0,4.0), (0.0,1.0)])
Data_output1 = mapslices(Ammonia_oxidation_rate, Data_input1; dims=2)
out_r_AO1 = [Data_input1 Data_output1]

# Generate LHC data using the second domain [0, 40] x [0, 9.1]
Data_LHC2 = randomLHC(10000, 2)
Data_input2 = scaleLHC(Data_LHC2, [(0.0,40.0), (0.0,9.1)])
Data_output2 = mapslices(Ammonia_oxidation_rate, Data_input2; dims=2)
out_r_AO2 = [Data_input2 Data_output2]

# Concatenate data and save to CSV
out_r_AO = [out_r_AO1; out_r_AO2]
writedlm("nitrification_training_data.csv", out_r_AO, ',')