using McCormick, EAGO, JuMP, NaNMath, CSV, DataFrames

include("worse_case_subsea_surrogate.jl")

#validity_surrogate_path = "validity_surrogate"
#performance_surrogate_path = "performance_surrogate"
#ANN_valid = surrogate_build_model(validity_surrogate_path)
#ANN_perform = surrogate_build_model(performance_surrogate_path)



API = 35
SG_G = 0.6
SG_W = 1.0
ρᵒ_W = 1000.0

P_well = 5.52*10^6 # Pa

k_GLS = 8.33*10^(-3)
L_GLS = 4.0        # Length (m) of LLS
R_GLS = 0.6        # Radius (m) of LLS
P_GLS = 4.0*10^6   # Pa

k_LLS = 1.67*10^(-4)
L_LLS = 5.0        # Length (m) of LLS
R_LLS = 0.8        # Radius (m) of LLS
P_LLS = 4*10^6     # Operating Pressure (Pa) of LLS
H_LLS = 0.6        # Liquid Level in the LLS (m)

C_v1 = 1.67*10^(-2)
C_v2 = 0.1675

SG_O = 141.5/(131/5 + API)

# Fixed relations from Control Value (V-1)
P_1 = P_well
P_2 = P_GLS
ξO1 = 0.4

# Fixed relations from Gas-liquid Separator
P_3 = P_2
ξ3 = [1.0; 0.0; 0.0]

# Fixed relations from Control Value (V-2)
P_5 = P_LLS

# Fixed relations from Liquid-liquid separator
V_LLS = L_LLS*((H_LLS - R_LLS)*sqrt(2.0*R_LLS*H_LLS - H_LLS^2) + (R_LLS^2)*acos(1.0 - H_LLS/R_LLS))
ξ6 = [0.0; 1.0; 0.0]
ξ8 = [1.0; 0.0; 0.0]
P_8 = P_LLS

ξG7_spec = 0.05

guard_tol = 1E-8

f(x) = -x[1]
function gSIP_perform(x, p)
    u1 = x[2]
    u2 = x[3]
    ξG1 = p[1]
    ξW1 = 1 - ξO1 - ξG1
    SG_mix_inv_1 = ξG1/SG_G + ξW1/SG_W + ξO1/SG_O
    sqrt_arg1 = positive((P_1 - P_2)*SG_mix_inv_1) + guard_tol
    m2 = u1*C_v1*sqrt(sqrt_arg1)
    ξG2 = ξG1
    ξW2 = ξW1
    HGLS, ξG4, P4, ρ4 = ANN_perform((m2, u2, ξG2, ξW2,))
    sqrt_arg2 = positive(ρᵒ_W*(P4 - P_LLS)/positive(ρ4)) + guard_tol
    m4 = u2*C_v2*sqrt(sqrt_arg2)
    ξG7 = ξG4*exp(-k_LLS*m4*V_LLS/(positive(ρ4)+ guard_tol))
    return ξG7 - ξG7_spec
end

function gSIP_valid(x, p)
    u1 = x[2]
    u2 = x[3]
    ξG1 = p[1]
    ξW1 = 1 - ξO1 - ξG1
    SG_mix_inv_1 = ξG1/SG_G + ξW1/SG_W + ξO1/SG_O
    sqrt_arg1 = positive((P_1 - P_2)*SG_mix_inv_1) + guard_tol
    m2 = u1*C_v1*sqrt(sqrt_arg1)
    ξG2 = ξG1
    ξW2 = ξW1
    valid_flout = ANN_valid((m2, u2, ξG2, ξW2,))
    0.01*(1.0 - 1.0*valid_flout[1])
end

function gSIP(x,p)
    g1 = gSIP_valid(x, p)
    g2 = gSIP_perform(x, p)
    return x[1] - max(g1, g2)
end
function solve_worst_case_sip()

    x_l = [-0.05, 0.35, 0.35]
    x_u = [0.05, 0.8, 0.8]
    p_l = [0.35]
    p_u = [0.5]

    opt = EAGO.Optimizer

    sip_tol = 1E-4
    abs_tol = sip_tol/10.0
    rel_tol = sip_tol/10.0

    sip_result = explicit_sip_solve(x_l, x_u, p_l, p_u, f, [gSIP],
                                    sip_absolute_tolerance = 1E-3,
                                    sip_relative_tolerance = 1E-3,
                                    sip_optimizer = opt,
                                    absolute_tolerance = abs_tol,
                                    relative_tolerance = rel_tol,
                                    #sip_verbosity = 4,
                                    sip_sense = :min,
                                    verbosity = 1,
                                    output_iterations = 10000,
                                    upper_bounding_depth = 50)
    return sip_result
end

start_time = time_ns()
worst_case_sip_result = solve_worst_case_sip()
end_time = time_ns()
elapsed_time = end_time - start_time
@show elapsed_time/(1000^3)
