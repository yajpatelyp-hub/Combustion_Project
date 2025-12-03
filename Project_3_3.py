import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp


# =============================================================================
# Helper: Normalize mass fractions safely
# =============================================================================
def normalize_mass_fractions(Y):
    Y = np.array(Y, dtype=float)

    # If anything is non-finite, fall back to uniform
    if not np.all(np.isfinite(Y)):
        return np.ones_like(Y) / len(Y)

    Y = np.clip(Y, 0.0, None)
    s = Y.sum()

    if (not np.isfinite(s)) or (s <= 0.0):
        return np.ones_like(Y) / len(Y)
        
    return Y / s


# =============================================================================
# PFR SOLVER #1: Verification Solver
# Used for:
#   - Constant h (analytic test)
#   - Constant cp (analytic test)
#   - No heat transfer (h=0) case
#   - Comparison to batch reactors (Verification 4.2, 4.3)
# =============================================================================
def solve_pfr_cantera(
    mechfile,
    X0,
    T0,
    p0,
    mdot,
    D,
    Tw,
    x_end,
    Npts=400,
    h_correlation="Dittus-Boelter",
    h_override=None,
    cp_override=None
):


    gas = ct.Solution(mechfile)
    gas.TPX = T0, p0, X0

    # Geometry
    A = np.pi * D**2 / 4.0
    Per = np.pi * D
    ns = gas.n_species

    # Initial vector
    Y0 = gas.Y.copy()
    y0 = np.concatenate(([T0], Y0))

    x_eval = np.linspace(0, x_end, Npts)

    # -------------------------------------------------------------------------
    # Heat-transfer correlation
    # -------------------------------------------------------------------------
    def h_conv_func(T, Y, rho, u):
        if h_override is not None:
            return h_override(T, Y, p0, rho, u)

        gas.TPY = T, p0, Y
        mu = gas.viscosity
        k = gas.thermal_conductivity
        cp_mass = gas.cp_mass
        Pr = cp_mass * mu / k

        Re = rho * u * D / mu

        if Re < 2300:
            Nu = 3.66
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4

        return Nu * k / D

    # -------------------------------------------------------------------------
    # RHS
    # -------------------------------------------------------------------------
    def rhs(x, y):
        T = float(y[0])
        T = max(300.0, min(4000.0, T))
        
        Y = normalize_mass_fractions(y[1:])

        gas.TPY = T, p0, Y

        rho = gas.density
        cp = gas.cp_mass if cp_override is None else cp_override(T)
        h_i = gas.partial_molar_enthalpies / gas.molecular_weights
        u = mdot / (rho * A)

        # reaction rates
        omega_molar = gas.net_production_rates
        omega_mass = omega_molar * (gas.molecular_weights / 1000.0)

        dYdx = omega_mass / (rho * u)

        # heat transfer
        h_conv = h_conv_func(T, Y, rho, u)

        Tw_val = Tw(x) if callable(Tw) else Tw

        conv_term = h_conv * Per / (mdot * cp) * (Tw_val - T)
        chem_term = -(1.0 / (rho * u * cp)) * np.dot(h_i, omega_mass)

        dTdx = conv_term + chem_term

        return np.concatenate(([dTdx], dYdx))

    sol = solve_ivp(rhs, (0, x_end), y0, t_eval=x_eval,
                    method="BDF", atol=1e-12, rtol=1e-8)

    if not sol.success:
        raise RuntimeError("ODE integration failed: " + sol.message)

    # Convert back to mole fractions
    T_sol = sol.y[0]
    Y_sol = sol.y[1:]
    X_sol = np.zeros_like(Y_sol)

    for j, x in enumerate(x_eval):
        gas.TPY = T_sol[j], p0, Y_sol[:, j]
        X_sol[:, j] = gas.X

    return {
        "x": x_eval,
        "T": T_sol,
        "Y": Y_sol,
        "X": X_sol,
        "gas": gas,
        "mech": mechfile,
        "mdot": mdot
    }


# =============================================================================
# PFR SOLVER #2: Methane Pyrolysis Solver
# Used for:
#   - Part 5 of project
#   - Mechanism sensitivity study
#   - High-T CH4 cracking with variable h
# =============================================================================
def solve_pfr_pyrolysis(
    mechfile,
    X0,
    T0,
    p0,
    u0,
    D,
    Tw,
    x_end,
    Npts=400
):


    gas = ct.Solution(mechfile)
    gas.TPX = T0, p0, X0
    ns = gas.n_species

    # Geometry
    A = np.pi * D**2 / 4.0
    Per = np.pi * D

    # Compute mdot from inlet velocity
    rho0 = gas.density
    mdot = rho0 * A * u0

    # Initial vector
    Y0 = gas.Y.copy()
    y0 = np.concatenate(([T0], Y0))
    x_eval = np.linspace(0, x_end, Npts)


    # -------------------------------------------------------------------------
    # Heat transfer correlation
    # -------------------------------------------------------------------------
    def h_conv(T, Y, p):
        gas.TPY = T, p, Y
        rho = gas.density
        mu  = gas.viscosity
        cp  = gas.cp_mass
        k   = gas.thermal_conductivity
        u   = mdot / (rho * A)

        Re = rho * u * D / mu
        Pr = cp * mu / k

        if Re < 2300:
            Nu = 3.66
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4

        return Nu * k / D


    # -------------------------------------------------------------------------
    # RHS
    # -------------------------------------------------------------------------
    def rhs(x, y):
        T = y[0]
        Y = normalize_mass_fractions(y[1:])

        gas.TPY = T, p0, Y

        rho = gas.density
        cp  = gas.cp_mass
        h_i = gas.partial_molar_enthalpies / gas.molecular_weights
        u   = mdot / (rho * A)

        # reaction rates
        omega_molar = gas.net_production_rates
        omega_mass  = omega_molar * (gas.molecular_weights / 1000.0)

        dYdx = omega_mass / (rho * u)

        # dynamic heat-transfer coefficient
        h = h_conv(T, Y, p0)

        Tw_val = Tw(x) if callable(Tw) else Tw

        conv_term = h * Per / (mdot * cp) * (Tw_val - T)
        chem_term = -(1.0/(rho*u*cp)) * np.dot(h_i, omega_mass)

        dTdx = conv_term + chem_term

        return np.concatenate(([dTdx], dYdx))


    # Integrate
    sol = solve_ivp(rhs, (0, x_end), y0, t_eval=x_eval,
                    method="BDF", atol=1e-10, rtol=1e-8)

    if not sol.success:
        raise RuntimeError(sol.message)

    T_sol = sol.y[0]
    Y_sol = sol.y[1:]

    # Convert to mole fractions
    X_sol = np.zeros_like(Y_sol)
    for j in range(len(x_eval)):
        gas.TPY = T_sol[j], p0, Y_sol[:, j]
        X_sol[:, j] = gas.X

    return {
        "x": x_eval,
        "T": T_sol,
        "Y": Y_sol,
        "X": X_sol,
        "gas": gas,
        "mdot": mdot,
        "mech": mechfile
    }