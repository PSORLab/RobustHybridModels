using DataFrames, CSV, McCormick, EAGO, JuMP

include("nitrification_surrogate.jl")

# alternatively McCormick can be used instead EAGO in this example
include("nitrification_surrogate.jl")

function Ammonia_oxidation_rate(x)
    V = 1000.0
    m_in = V / 240 # L/s
    m_out = V / 240 # L/s
    X_AO = 0.505 # g VSS/L
    M_N = 14.0067 #
    c_NH = x[1]
    c_O = x[2]
    r_AOmax = 0.67 # mg N-NH+ /g VSS_AO min
    K_OAO = 0.3 # mg O2/L
    K_SAO = 0.34 # mg N-NH4+/L
    K_IAO = 6200 # mg N-NH4+/L
    r_AO = r_AOmax*(c_NH/(K_SAO + c_NH + c_NH^2/K_IAO))*(c_O/(K_OAO + c_O)); # mg N-NH+ /g VSS_AO min
    return r_AO
end

## Kinetics Model
function DataModel(t, y, p, u)

    ## Parameters
    K_SAO = 0.24 # mg N-NH/L
    K_IAO = 6200.0 # mg N-NH/L
    K_OAO = 0.3 # mg/L
    K_SNO = 1.6 # mg N-NI/L
    K_INO = 13000.0 # mg N-NI/L
    K_ONO = 1.5 # mg/L
    X_AO = 0.505 # g VSS/L
    X_NO = 0.151 # g VSS/L
    rNO_MAX = 1.07/60 # mg N-NI/(g-VSS_NO*s)

    # Volumetric flow
    V = 1000.0 # L
    m_in = V/240 # L/s
    m_out = V/240 # L/s
    Cs = 31.0 # mg N-NH/L

    # Oxygen flow
    SOTE = 0.1
    cO_s = 9.1 # mg/L
    Phi_OAO = 2.5 # mg O2/ mg N-NH
    Phi_ONO = 0.32 # mg O2/ mg N-NI

    cNH, cNI, cNA, cO = y

    if t >= 3080.0 && t <= 3100.0
        Cin = p[1]
    else
        Cin = Cs
    end

    # ANN
    #@show cNH, cO
    rAO = ANN1([cNH, cO])/60.0 # mg N-NH/(g-VSS_AO*s)
    #rAO = Ammonia_oxidation_rate([cNH, cO])
    t1 =(cNI/positive(K_SNO + cNI + cNI^2 /K_INO))
    t2 =(cO/positive(K_ONO + cO))
    rNO = rNO_MAX*t1*t2

    # Reaction Rate
    R_NH = -rAO*X_AO  # mg N-NH/(L*s)
    R_NI = rAO*X_AO - rNO*X_NO  # mg N-NI/(L*s)
    R_NA = rNO*X_NO  # mg N-NA/(L*s)

    W = 0.2967*u[1] # mg/s
    SOTR = SOTE*W # mg/s
    kla = SOTR/cO_s/V # s^(-1)
    R_O = kla*(cO_s - cO) - rAO*Phi_OAO*X_AO - rNO*Phi_ONO*X_NO # mg O2/(L*s)

    # Mass Balacne
    h1 = 1 /V*(m_in*Cin - m_out*cNH) + R_NH
    h2 = R_NI
    h3 = R_NA
    h4 = R_O

    return h1, h2, h3, h4
end

t0 = 0.0
h = 20.0
tspan_0 = 3080.0
n_out_0 = convert(Int, round(tspan_0 / h))

function EE_Form_Initial(u0)
    cNH0 = 30.0
    cNI0 = 0.0
    cNA0 = 0.0
    cO0 = 0.0
    y0 = cNH0, cNI0, cNA0, cO0
    p0 = 40.0
    y = y0
    y1, y2, y3, y4 = y0
    t = t0
    for j = 1:n_out_0
        t = t + h
        dy1, dy2, dy3, dy4 = DataModel(t, y, p0, u0)    # Perform Step
        #y1 = bnd(y1 + h*dy1, 0.0, 40.0)  # Enforce physical bounds on quantity at each step
        #y2 = bnd(y2 + h*dy2, 0.0, 40.0)  # Enforce physical bounds on quantity at each step
        #y3 = bnd(y3 + h*dy3, 0.0, 40.0)  # Enforce physical bounds on quantity at each step
        #y4 = bnd(y4 + h*dy4, 0.0, 9.1)
        y1 = y1 + h*dy1
        y2 = y2 + h*dy2
        y3 = y3 + h*dy3
        y4 = y4 + h*dy4
        y = y1, y2, y3, y4              # Repackage as tuple
    end
    return t, y
end

u0 = 440.0
tc, yc = EE_Form_Initial(u0)
# 500, 50 bad....
#h = 0.1
tspan = 500.0
n_out = convert(Int, round(tspan / h))
function EE_Form(p, u)
    y1, y2, y3, y4 = yc
    y = y1, y2, y3, y4
    t = tc
    for j = 1:n_out
            t = t + h
            #@show j
            dy1, dy2, dy3, dy4 = DataModel(t, y, p, u)
            #y1 = bnd(y1 + h*dy1, 0.0, 40.0)  # Enforce physical bounds on quantity at each step
            #y2 = bnd(y2 + h*dy2, 0.0, 40.0)  # Enforce physical bounds on quantity at each step
            #y3 = bnd(y3 + h*dy3, 0.0, 40.0)  # Enforce physical bounds on quantity at each step
            #y4 = bnd(y4 + h*dy4, 0.0, 9.1)
            #y1 = positive(y1 + h*dy1)
            #y2 = positive(y2 + h*dy2)
            #y3 = positive(y3 + h*dy3)
            #y4 = positive(y4 + h*dy4)
            y1 = y1 + h*dy1
            y2 = y2 + h*dy2
            y3 = y3 + h*dy3
            y4 = positive(y4 + h*dy4)
            y = y1, y2, y3, y4              # Repackage as tuple
            #@show y
    end
    return t, y
end

#=
function gsip(p, u)
    @show p, u
    t, y = EE_Form(p, u)
    cNH_f, cNI_f, cNA_f, cO_f = y
    @show cNH_f - SP
    return cNH_f - SP
end
=#
function gsip(u,p)
    SP = 30.0
    y1, y2, y3, y4 = yc
    y = y1, y2, y3, y4
    t = tc
    for j = 1:n_out
        t = t + h
        dy1, dy2, dy3, dy4 = DataModel(t, y, p, u)
        y1 = y1 + h*dy1
        y2 = y2 + h*dy2
        y3 = y3 + h*dy3
        y4 = positive(y4 + h*dy4)
        y = y1, y2, y3, y4              # Repackage as tuple
    end
    cNH_f, cNI_f, cNA_f, cO_f = y
    return cNH_f - SP
end

f(u) = tspan * u[1]

function solve_nitrification_sip()

    sip_x_lo = Float64[440.0]
    sip_x_hi = Float64[2000.0]
    sip_p_lo = Float64[31.0]
    sip_p_hi = Float64[40.0]

    opt = EAGO.Optimizer
    sip_result = explicit_sip_solve(
        sip_x_lo,
        sip_x_hi,
        sip_p_lo,
        sip_p_hi,
        f,
        [gsip],
        sip_absolute_tolerance = 3000.00, # a little less than 1E-2 rel tolerance (but not significant)
        sip_optimizer = opt,
        upper_bounding_depth = 12,
        verbosity = 0
    )
    return sip_result
end

polymer_sip_result = solve_nitrification_sip()
