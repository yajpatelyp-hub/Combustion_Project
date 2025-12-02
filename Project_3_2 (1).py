import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp

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
    h_override=None,       # NEW
    cp_override=None       # NEW (optional)
):

    gas = ct.Solution(mechfile)

    # Initialize gas state
    gas.TPX = T0, p0, X0
    Y0 = gas.Y.copy()

    # Geometry
    A = np.pi * D**2 / 4
    Per = np.pi * D

    ns = gas.n_species

    # Initial state vector
    y0 = np.concatenate(([T0], Y0))

    # x-grid for solution sampling
    x_eval = np.linspace(0, x_end, Npts)

    # -------------------------------------------------------------------------
    # Heat transfer correlation (with fallback and overrides)
    # -------------------------------------------------------------------------
    def h_conv_func(T, Y, rho, u):
        # If user overrides h (for analytic test), use that
        if h_override is not None:
            return h_override(T, Y, p0, rho, u)

        # Otherwise compute using correlations
        gas.TPY = T, p0, Y
        mu = gas.viscosity
        k = gas.thermal_conductivity
        cp_mass = gas.cp_mass

        # FIXED BUG: correct Pr definition
        Pr = cp_mass * mu / k

        Re = rho * u * D / mu

        # Laminar / turbulent logic
        if Re < 2300:
            Nu = 3.66                # Laminar fully developed
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4  # Dittus-Boelter

        return Nu * k / D

    # -------------------------------------------------------------------------
    # RHS of ODE
    # -------------------------------------------------------------------------
    def rhs(x, y):
        T = y[0]
        Y = np.maximum(y[1:], 0)

        # Normalize Y
        s = Y.sum()
        if s == 0:
            Y[:] = 1.0/ns
        else:
            Y = Y/s

        # Update gas state
        gas.TPY = T, p0, Y

        rho = gas.density
        cp = gas.cp_mass if cp_override is None else cp_override(T)
        h_i = gas.partial_molar_enthalpies / gas.molecular_weights
        u = mdot / (rho * A)

        # Reaction rates
        omega_molar = gas.net_production_rates
        omega_mass = omega_molar * (gas.molecular_weights / 1000.0)

        # Species ODE
        dYdx = omega_mass / (rho * u)

        # Heat transfer coef
        h_conv = h_conv_func(T, Y, rho, u)

        # Wall temperature can be a function
        Tw_val = Tw(x) if callable(Tw) else Tw

        # Temperature ODE
        conv_term = h_conv * Per / (mdot * cp) * (Tw_val - T)
        chem_term = -(1.0/(rho*u*cp)) * np.dot(h_i, omega_mass)
        dTdx = conv_term + chem_term

        return np.concatenate(([dTdx], dYdx))

    # Solve ODE
    sol = solve_ivp(rhs, (0, x_end), y0, t_eval=x_eval,
                    method="BDF", atol=1e-12, rtol=1e-8)

    if not sol.success:
        raise RuntimeError(sol.message)

    T_sol = sol.y[0]
    Y_sol = sol.y[1:]
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
        "mech": mechfile,
        "mdot": mdot,
        "D": D,
        "Tw": Tw,
        "p0": p0
    }