using DataFrames, CSV, McCormick, EAGO, JuMP

include("polymerization_surrogate.jl")

# Input parameters
R = 8.345 # kJ/kmol/K
Z_P0 = 4.9167e5 # m^3/kmol/s
E_P0 = 1.8283e4 # kJ/kmol
Z_xip = 3.0233e13 # s^-1
E_xip = 1.17e5 # kJ/kmol
rho_p = 1.2e3 # kg/m^3
Mm = 1.0012e2 # kg/kmol
rho_m = 9.151e2 # kg/m^3
Ms = 9.214e1 # kg/kmol
rho_s = 8.42e2 # kg/m^3
Vs0 = 4.5e-4 # m^3
T_gp = 3.872e2 # K
B = 0.03
Z_fm = 4.661e9 # m^3/kmol/s
E_fm = 7.4479e4 # kJ/kmol
Zi = 1.0533e15 # s^-1
Ei = 1.2877e5 # kJ/kmol

# loading
Ci0 = 0.13 # kmol/m^3
Cm0 = 6.01 # kmol/m^3
V0 = 4.5e-4 / 0.3 # m^3

# Input other parameters
DelH_P = 5.78e4 # kJ/kmol
c = 2.2 # kJ/kg/K
m = 1.257 # kg
a1_0 = 0.0038 # s^-1
a2_0 = 0.0008 # s^-1
a3 = 0.00037 # s^-1
a4 = 0.0664 # K/kJ
T_cw = 2.797e2 # K
T_inf = 2.932e2 # K
cw = 4.2 # kJ/kg/K
rho_w = 1e3 # kg/m^3
P_max = 3.13 # kJ/s
Fcw_max = 2.55e-4 # m^3/s
Cm0 = 6.01 # kmol/m^3
phi_m0 = Cm0*Mm/rho_m
eps0 = phi_m0*(rho_m/rho_p - 1.0)
a0 = DelH_P*V0*(1.0 + eps0) / m / c

V0 = 4.5e-4 / 0.3 # m^3
Cs0 = rho_s*Vs0/Ms/V0

h = 5.0
i_out = 12
Tadiabatic = 4000.0

function rate_law(Cm, Ci, T)
    mu1 = Mm/(1.0 + eps0)*(Cm0 - Cm)
    Cs = Cs0*(1.0 + eps0*(Cm/Cm0))/(1.0 + eps0)
    phi_p = (mu1/rho_p)/(mu1/rho_p + Cm*Mm/rho_m + Cs*Ms/rho_s)   # Can be strengthened using a sum-div form

    A = 0.168 - 8.21e-6*(T - T_gp)^2
    D = exp(2.3*(1.0 - phi_p)/(A + B*(1.0 - phi_p)))

    # ANN
    xi0 = ANN1((Cm, Ci, T,))

    k_P0 = Z_P0*arh(T, E_P0/R)                                     # Used arh(x,k) = exp(-k,x) which overloads to envelope.
    k_xip = Z_xip*arh(T, E_xip/R)
    kP = k_P0/(1.0 + xi0*k_P0/(D*k_xip))                           # Could strengthened D*k_xip using exp(x)*y, Khajavirad form envelope
                                                                   # A stronger relaxation for x/(a + x*y) would be of interest here.
    k_fm = Z_fm*arh(T, E_fm/R)
    ki = Zi*arh(T, Ei/R)

    # Rate law
    Rm = -Cm*xi0*(kP + k_fm)
    Ri = -ki*Ci
    return Rm, Ri, xi0, kP
end

function dynamics(y, P, Fcw)
    Cm, Ci, T, Tj = y
    Rm, Ri, xi0, kP = rate_law(Cm, Ci, T)
    xm = (1.0 - Cm/Cm0)/(1.0 + eps0*Cm/Cm0)                                    # Can add envelope to strengthen...
                                                                               # Unlikely to be able to strengthen further based on non-negativity assumptions

    a1 = a1_0*(1.0 + eps0*xm)*(0.2 + 0.8*exp(-7*xm^3))                         # Just a function of xm again can strengthen with envelope...
    a2 = a2_0*(1.0 + eps0*xm)*(0.2 + 0.8*exp(-7*xm^3))

    P_coord, Fcw_coord = P, Fcw                                                # No coordination rule

    F1 = (1.0 + eps0*Cm/Cm0)*Rm                                                # Rm is -Cm... can explicit write this and reduce Cm*Cm overestimation
    F2 = Ri + eps0*Ci/Cm0*Rm
    F3 = a0*kP*xi0*Cm/(1.0 + eps0*Cm/Cm0) + a1*(Tj - T)                        # Explicit treatment could tighten Cm/(1.0 + eps0*Cm/Cm0) subexpression
    F4 = a2*(T - Tj) + a3*(T_inf - Tj) + a4*(P_coord - Fcw_coord*rho_w*cw*(Tj - T_cw))
    return F1, F2, F3, F4
end

t0 = 0.0
Cm0 = 6.01 # kmol/m^3
Ci0 = 0.13 # kmol/m^3
T0 = 319.2
Tj0 = 319.2
y0 = Cm0, Ci0, T0, Tj0
n_out_0 = 240

function EE_Form_Initial(p1_0, p2_0)
    y = y0
    t = t0
    for j = 1:n_out_0
        for i = 1:i_out
            t = t + h
            y1, y2, y3, y4 = y .+ (h .* dynamics(y, p1_0, p2_0))    # Perform Step
            y1 = bnd(y1, 0.0, Cm0)                              # Enforce physical bounds on quantity at each step
            y2 = bnd(y2, 0.0, Ci0)
            y3 = bnd(y3, 279.7, Tadiabatic)
            y4 = bnd(y4, 279.7, Tadiabatic)
            y = y1, y2, y3, y4                                  # Repackage as tuple
        end
    end
    return t, y
end

p1_0 = 1.2
p2_0 = 0.65e-5
tc, yc = EE_Form_Initial(p1_0, p2_0)

n_out = 1
function EE_Form(p1, p2)
    t = tc
    y = yc
    for j = 1:n_out
        for i = 1:i_out
            t = t + h
            y1, y2, y3, y4 = y .+ (h .* dynamics(y, p1, p2))    # Perform Step
            y1 = bnd(y1, 0.0, Cm0)                              # Enforce physical bounds on quantity at each step
            y2 = bnd(y2, 0.0, Ci0)
            y3 = bnd(y3, 279.7, Tadiabatic)
            y4 = bnd(y4, 279.7, Tadiabatic)
            y = y1, y2, y3, y4                                  # Repackage as tuple
        end
    end
    return y
end

function gsip(x,p)
    y1, y2, Tf, y4, = EE_Form(p[1], p[2])
    return x[2] - (Tf - x[1])
end

f(x) = x[2]

function solve_polymerization_sip()

    sip_x_lo = Float64[368.0; -100.0]
    sip_x_hi = Float64[372.0; 20.0]
    sip_p_lo = Float64[0.0; 0.0]
    sip_p_hi = Float64[3.13; 2.55e-4]

    opt = EAGO.Optimizer

    sip_tol = 1E-1
    abs_tol = sip_tol/10.0
    rel_tol = sip_tol/10.0

    sip_result = explicit_sip_solve(sip_x_lo, sip_x_hi, sip_p_lo, sip_p_hi, f, [gsip],
                                    sip_absolute_tolerance = sip_tol,
                                    sip_optimizer = opt,
                                    absolute_tolerance = abs_tol,
                                    relative_tolerance = rel_tol,
                                    sip_verbosity = 4,
                                    sip_sense = :max,
                                    verbosity = 1)
    return sip_result
end

start_time = time_ns()
polymer_sip_result = solve_polymerization_sip()
end_time = time_ns()
elapsed_time = end_time - start_time
@show elapsed_time/(1000^3)
