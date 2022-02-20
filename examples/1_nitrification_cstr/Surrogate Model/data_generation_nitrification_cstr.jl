using JuMP, ForwardDiff, Ipopt
using EAGO
using IntervalArithmetic
using DelimitedFiles
using Statistics
using LatinHypercubeSampling

function Ammonia_oxidation_rate(x)
    c_NH = x[1]
    c_O = x[2]
    r_AOmax = 0.67 # mg N-NH+ /g VSS_AO min
    K_OAO = 0.3 # mg O2/L
    K_SAO = 0.34 # mg N-NH4+/L
    K_IAO = 6200 # mg N-NH4+/L
    r_AO = r_AOmax*(c_NH/(K_SAO + c_NH + c_NH^2/K_IAO))*(c_O/(K_OAO + c_O)); # mg N-NH+ /g VSS_AO min
    return r_AO
end

##
#Data_LHC = randomLHC(100000,2)
#Data_input = scaleLHC(Data_LHC,[(0.0,40.0),(0.0,9.1)])
#Data_output = mapslices(Ammonia_oxidation_rate,Data_input; dims=2)
#out_r_AO = [Data_input Data_output]
#writedlm("TrainingData.csv",out_r_AO,',')

Data_LHC1 = randomLHC(10000,2)
Data_input1 = scaleLHC(Data_LHC1,[(0.0,4.0),(0.0,1.0)])
Data_output1 = mapslices(Ammonia_oxidation_rate, Data_input1; dims=2)
out_r_AO1 = [Data_input1 Data_output1]

Data_LHC2 = randomLHC(10000,2)
Data_input2 = scaleLHC(Data_LHC2,[(0.0,40.0),(0.0,9.1)])
Data_output2 = mapslices(Ammonia_oxidation_rate, Data_input2; dims=2)
out_r_AO2 = [Data_input2 Data_output2]

out_r_AO = [out_r_AO1; out_r_AO2]

writedlm("TrainingData2.csv",out_r_AO,',')
