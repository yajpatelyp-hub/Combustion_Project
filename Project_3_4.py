import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp


# =============================================================================
# Utility: Clip mass fractions but DO NOT normalize inside ODE
# =============================================================================
def clip_mass_fractions(Y):
    """Clip negative numerical noise, but do NOT normalize."""
    Y = np.array(Y, dtype=float)
    Y[Y < 0] = 0.0
    return Y


# =============================================================================
# Solver #1 — Verification PFR Solver (used for Part 4)
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
    h_override=None,
    cp_override=None
):
    """
    General PFR ODE solver used for Verification 4.1–4.3.
    """

    gas = ct.Solution(mechfile)
    gas.TPX = T0, p0, X0
    ns = gas.n_species

    # Geometry
    A = np.pi * D**2 / 4.0
    Per = np.pi * D

    # Initial state vector
    Y0 = gas.Y.copy()
    y0 = np.concatenate(([T0], Y0))
    x_eval = np.linspace(0.0, x_end, Npts)

    # -------------------------------------------------------------------------
    # Heat transfer model
    # -------------------------------------------------------------------------
    def h_conv(T, Y, rho, u):
        # If user forces constant h (verification 4.1), override here
        if h_override is not None:
            return h_override(T, Y, p0, rho, u)

        gas.TPY = T, p0, Y
        mu = gas.viscosity
        k = gas.thermal_conductivity
        cp = gas.cp_mass

        Re = rho * u * D / mu
        Pr = cp * mu / k

        if Re < 2300:
            Nu = 3.66
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4

        return Nu * k / D

    # -------------------------------------------------------------------------
    # RHS of ODE
    # -------------------------------------------------------------------------
    def rhs(x, y):
        T = float(y[0])
        T = max(T, 10.0)

        # CLIP ONLY — NO NORMALIZATION
        Y = clip_mass_fractions(y[1:])

        gas.TPY = T, p0, Y
        rho = gas.density
        cp = gas.cp_mass if cp_override is None else cp_override(T)
        h_i = gas.partial_molar_enthalpies / gas.molecular_weights
        u = mdot / (rho * A)

        # Reaction rates
        omega_molar = gas.net_production_rates
        omega_mass = omega_molar * (gas.molecular_weights / 1000.0)

        dYdx = omega_mass / (rho * u)

        # Heat transfer coefficient
        h = h_conv(T, Y, rho, u)

        # Wall temperature (constant or function)
        Tw_val = Tw(x) if callable(Tw) else Tw

        # Energy equation
        conv_term = h * Per / (mdot * cp) * (Tw_val - T)
        chem_term = -(1.0 / (rho * u * cp)) * np.dot(h_i, omega_mass)

        dTdx = conv_term + chem_term

        return np.concatenate(([dTdx], dYdx))

    # -------------------------------------------------------------------------
    # Integrate ODEs
    # -------------------------------------------------------------------------
    sol = solve_ivp(rhs, (0.0, x_end), y0, t_eval=x_eval,
                    method="BDF", atol=1e-12, rtol=1e-8)

    if not sol.success:
        raise RuntimeError("ODE integration failed:\n" + sol.message)

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


# =============================================================================
# Solver #2 — Methane Pyrolysis Solver (Problem 5)
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
    """
    High-temperature methane pyrolysis PFR solver.
    """

    gas = ct.Solution(mechfile)
    gas.TPX = T0, p0, X0
    ns = gas.n_species

    # Geometry
    A = np.pi * D**2 / 4.0
    Per = np.pi * D

    # Compute mdot from inlet velocity
    rho0 = gas.density
    mdot = rho0 * A * u0

    # Initial state vector
    Y0 = gas.Y.copy()
    y0 = np.concatenate(([T0], Y0))
    x_eval = np.linspace(0.0, x_end, Npts)

    # -------------------------------------------------------------------------
    # Heat transfer (Nu correlation)
    # -------------------------------------------------------------------------
    def h_conv(T, Y, p):
        gas.TPY = T, p, Y
        rho = gas.density
        mu = gas.viscosity
        cp = gas.cp_mass
        k = gas.thermal_conductivity
        u = mdot / (rho * A)

        Re = rho * u * D / mu
        Pr = cp * mu / k

        if Re < 2300:
            Nu = 3.66
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4

        return Nu * k / D

    # -------------------------------------------------------------------------
    # RHS for pyrolysis
    # -------------------------------------------------------------------------
    def rhs(x, y):
        T = float(y[0])
        T = max(T,10.0)

        T_sol = sol.y[0]
        Y_sol = sol.y[1:]

        gas.TPY = T, p0, Y
        rho = gas.density
        cp = gas.cp_mass
        h_i = gas.partial_molar_enthalpies / gas.molecular_weights
        u = mdot / (rho * A)

        # Reaction rates
        omega_molar = gas.net_production_rates
        omega_mass = omega_molar * (gas.molecular_weights / 1000.0)

        dYdx = omega_mass / (rho * u)

        # Dynamic heat transfer
        h = h_conv(T, Y, p0)
        Tw_val = Tw(x) if callable(Tw) else Tw

        conv_term = h * Per / (mdot * cp) * (Tw_val - T)
        chem_term = -(1.0 / (rho * u * cp)) * np.dot(h_i, omega_mass)

        dTdx = conv_term + chem_term

        return np.concatenate(([dTdx], dYdx))

    # -------------------------------------------------------------------------
    # Integrate
    # -------------------------------------------------------------------------
    sol = solve_ivp(rhs, (0.0, x_end), y0, t_eval=x_eval,
                    method="BDF", atol=1e-10, rtol=1e-8)

    if not sol.success:
        raise RuntimeError("ODE integration failed:\n" + sol.message)

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