# Copyright (c) 2021: Matthew Wilhelm, Chenyu Wang, Matthew Stuber, 
# and the University of Connecticut (UConn).
# This code is licensed under the MIT license (see LICENSE.md for full details).
#################################################################################
# RobustHybridModels
# Examples from the paper Semi-Infinite Optimization with Hybrid Models
# https://github.com/PSORLab/RobustHybridModels
#################################################################################
# examples/1_nitrification_cstr/nitrification_sip.jl
# This file contains the SIPs for Example 1 in the paper. The feasibility problem
# and operation problem are both solved using the ANN for the ammonia oxidation
# rate given in examples/1_nitrification_cstr/nitrification_surrogate.jl.
#################################################################################

using CSV, DataFrames, UnPack, DifferentialEquations, EAGO

include("nitrification_surrogate.jl")

# Dictionary of parameters to pass through each function
AOR_PARAMETERS = Dict{Symbol,Any}()
AOR_PARAMETERS[:V]       = 1000.0    # L
AOR_PARAMETERS[:m_in]    = 4.167     # L/s
AOR_PARAMETERS[:m_out]   = 4.167     # L/s
AOR_PARAMETERS[:C_NHsat] = 31.0      # mg N-NH4+/L
AOR_PARAMETERS[:C_Osat]  = 9.1       # mg O2/L
AOR_PARAMETERS[:X_AO]    = 0.505     # mg VSS_AO/L
AOR_PARAMETERS[:r_AOmax] = 0.67/60   # mg N-NH4+/(g VSS_AO*s)
AOR_PARAMETERS[:Psi_OAO] = 2.5       # mg O2/mg N-NH4+
AOR_PARAMETERS[:K_SAO]   = 0.24      # mg N-NH4+/L
AOR_PARAMETERS[:K_IAO]   = 6200.0    # mg N-NH4+/L
AOR_PARAMETERS[:K_OAO]   = 0.3       # mg/L
AOR_PARAMETERS[:X_NO]    = 0.151     # mg VSS_NO/L
AOR_PARAMETERS[:r_NOmax] = 1.07/60   # mg N-NO2-/(g VSS_NO*s)
AOR_PARAMETERS[:Psi_ONO] = 0.32      # mg O2/mg N-NO2-
AOR_PARAMETERS[:K_SNO]   = 1.6       # mg N-NO2-/L
AOR_PARAMETERS[:K_INO]   = 13000.0   # mg N-NO2-/L
AOR_PARAMETERS[:K_ONO]   = 1.5       # mg/L
AOR_PARAMETERS[:SOTE]    = 0.1

# Kinetic model for the ammonia oxidation rate
function Ammonia_oxidation_rate(C_NH, C_O, params)
    @unpack r_AOmax, K_SAO, K_IAO, K_OAO = params
    r_AO = r_AOmax*(C_NH/(K_SAO + C_NH + C_NH^2/K_IAO))*(C_O/(K_OAO + C_O))   # mg N-NH4+/(g VSS_AO*s)
    return r_AO
end
Ammonia_oxidation_rate(C_NH, C_O) = Ammonia_oxidation_rate(C_NH, C_O, AOR_PARAMETERS)

# Kinetic model solved over a long period of time to get a steady-state solution
function rhs_ode!(dy, y, p, t, params)

    @unpack V, m_in, m_out, C_NHsat, C_Osat, X_AO, Psi_OAO, X_NO, r_NOmax, Psi_ONO, K_SNO, K_INO, K_ONO, SOTE = params

    C_NH, C_NI, C_NA, C_O = y
    C_O = positive(C_O)
    C_in = C_NHsat

    r_AO = Ammonia_oxidation_rate(C_NH, C_O)
    r_NO = r_NOmax*C_NI/(K_SNO + C_NI + C_NI^2/K_INO)*(C_O/(C_O + K_ONO))

    # Reaction rates
    R_NH = -r_AO*X_AO              # mg N-NH4+/(L*s)
    R_NI = r_AO*X_AO - r_NO*X_NO   # mg N-NO2-/(L*s)
    R_NA = r_NO*X_NO               # mg N-NO3-/(L*s)

    SOTR = SOTE*0.2967*p[1]                                            # mg/s
    kla = SOTR/C_Osat/V                                                # s^(-1)
    R_O = kla*(C_Osat - C_O) - r_AO*Psi_OAO*X_AO - r_NO*Psi_ONO*X_NO   # mg O2/(L*s)

    # Mass balances
    dy[1] = (1/V)*(m_in*C_in - m_out*C_NH) + R_NH
    dy[2] = R_NI
    dy[3] = R_NA
    dy[4] = R_O

    return dy
end
rhs_ode!(dy, y, p, t) = rhs_ode!(dy, y, p, t, AOR_PARAMETERS)

# Steady-state solution
alg = Tsit5()
prob = ODEProblem(rhs_ode!, [30.0; 0.0; 0.0; 0.0], (0, 3080))
sol = solve(prob, alg, reltol = 1E-8, abstol = 1E-8, p = [700])
steady_state_u0 = sol.u[end]

# Kinetic model solved in SIP
function rhs_ode_op(y, p, u, t, params)

    @unpack V, m_in, m_out, C_NHsat, C_Osat, X_AO, Psi_OAO, X_NO, r_NOmax, Psi_ONO, K_SNO, K_INO, K_ONO, SOTE = params

    C_NH, C_NI, C_NA, C_O = y
    C_in = (t <= 20.0) ? p[1] : C_NHsat

    r_AO = Ammonia_oxidation_ANN([C_NH, C_O])
    r_NO = r_NOmax*C_NI/(K_SNO + C_NI + C_NI^2/K_INO)*(C_O/(C_O + K_ONO))

    # Reaction rates
    R_NH = -r_AO*X_AO              # mg N-NH4+/(L*s)
    R_NI = r_AO*X_AO - r_NO*X_NO   # mg N-NO2-/(L*s)
    R_NA = r_NO*X_NO               # mg N-NO3-/(L*s)

    SOTR = SOTE*0.2967*u[1]                                            # mg/s
    kla = SOTR/C_Osat/V                                                # s^(-1)
    R_O = kla*(C_Osat - C_O) - r_AO*Psi_OAO*X_AO - r_NO*Psi_ONO*X_NO   # mg O2/(L*s)

    # Mass balances
    dy1 = 1/V*(m_in*C_in - m_out*C_NH) + R_NH
    dy2 = R_NI
    dy3 = R_NA
    dy4 = R_O

    return dy1, dy2, dy3, dy4
end
rhs_ode_op(y, p, u, t) = rhs_ode_op(y, p, u, t, AOR_PARAMETERS)

# Explicit Euler integration scheme for kinetic model solved in SIP
function EE_Form(p::S, u::T, u0) where {S,T}
    h = 10.0
    t = 0.0
    y1, y2, y3, y4 = u0
    y = y1, y2, y3, y4
    for i = 1:10
        dy1, dy2, dy3, dy4 = rhs_ode_op(y, p, u, t)
        t = t + h
        y1 = y1 + h*dy1
        y2 = y2 + h*dy2
        y3 = y3 + h*dy3
        y4 = y4 + h*dy4
        y = y1, y2, y3, y4
    end
    return y
end
EE_Form(p, u) = EE_Form(p, u, steady_state_u0)

# SIP constraint
function gSIP(x, p)
    y = EE_Form(p, x)
    return max(y[1] - 30.0, y[4] - 2.0)
end

# Feasibility problem
println("BEGIN SOLVING FEASIBILITY PROBLEM\n")
f_feasible(x) = x[2]
gSIP_feasible(x, p) = gSIP(x[1], p[1]) - x[2]
x_lo = Float64[440.0; -10.0]
x_hi = Float64[2000.0; 10.0]
p_lo = Float64[31.0]
p_hi = Float64[40.0]
r_feasibility = sip_solve(SIPRes(), x_lo, x_hi, p_lo, p_hi, f_feasible, Any[gSIP_feasible], res_sip_absolute_tolerance = 1E-3);
println("FINISH SOLVING FEASIBILITY PROBLEM\n")

# Operation problem
println("BEGIN SOLVING OPERATION PROBLEM\n")
f_operation(x) = -x[2]
gSIP_operation(x, p) = x[2] - gSIP(p[1], x[1])
x_lo = Float64[31.0; -10.0]
x_hi = Float64[40.0; 10.0]
p_lo = Float64[440.0]
p_hi = Float64[2000.0]
r_operation = sip_solve(SIPRes(), x_lo, x_hi, p_lo, p_hi, f_operation, Any[gSIP_operation], res_sip_absolute_tolerance = 1E-3);
println("FINISH SOLVING OPERATION PROBLEM\n")