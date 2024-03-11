import numpy as np

"""
    Mareile Wynants, 2022
    Ludwig Franzius Institute
    Leibniz University Hannover

    Translated from Matlab to python and refactored by Anton Kriese, 2023
"""

## Constants
# Inertia Coefficient Cm
Cm = 1.5

# Drag Coefficient Cd
Cd = 0.7

def wave_length(T: float, tol: float, d: float, g: float):
    """
        Function to calculate the general wave lenght (intermediate depth, also
        applicable for deep water and shallow water) depending on the wave period
        T, the water depth d and a tolerance tol [m]
    """

    L0 = g * T ** 2 / (2 * np.pi)
    Lalt = L0
    t = 1
    while t > tol:
        L = L0 * np.tanh(2 * np.pi / Lalt * d)
        t = abs(Lalt - L)
        Lalt = L

    return Lalt


def calculate_maximum_forces(D=6., rho=1025., d=28., T=12., H=8., g=9.81, n_zp=10, n_t=1000):
    """
    Implementation of the Morison Equation

    Valid for slender piles (D/L < 0.2)

        Inputs:
            - Pile diameter, D
            - Water density, rho
            - Water depth, d
            - Wave Period, T (in seconds)
            - Wave height, H
            - gravitational acceleration, g
            - Number of interpolation points, n_zp
            - Number of interpolated time steps, n_t
        Output:
            - The maximum force at each interpolated point, Ftmax
                Discretized concentrated loads along the pile length: Ftmax [N]
                Corresponding z-coordinates along the pile
                zppoint [m above MSL] (MSL = Mean Sea Level)
    """
    ## Spatial and temporal discretisation: Manual

    # Spatial increments zp over pile length
    # (starting at max. water surface elevation)
    zp = np.linspace(H/2, -d, n_zp)

    # pile segment centers
    zpm = (zp[1:] + zp[:-1]) / 2

    # n_t time stamps of one wave
    t = np.linspace(0, T, n_t)

    # Wave length L
    L = wave_length(T, 0.01, d, g)

    # Wave number k
    k = 2 * np.pi / L

    # Wave frequency w (omega)
    w = 2 * np.pi / T

    ## Morison: Inertia Force (Only for small D/L)
    # Horizontal particle acceleration x..
    x_dotdot = H/2 * w**2 * np.outer(np.cosh(k*(d+zpm)), np.sin(-w*t)) / np.sinh(k*d)

    # Inertia Force in N/m; (over time and depth increments)
    Fm = Cm * rho * np.pi * D**2/4 * x_dotdot

    ## Morison: Drag Force (for both small and large D/L)
    # Horizontal particle velocity x.
    x_dot = H/2 * w * np.outer(np.cosh(k*(d+zpm) / np.sinh(k*d)), np.cos(-w*t))

    # Drag Force in N/m; (over time and depth increments)
    Fd = np.multiply(Cd * rho * (D/2) * np.abs(x_dot), x_dot)

    ## Morison: Total Force
    # Total force over time = F(intertia) + F(drag); in N
    Ft = (Fm + Fd)

    # Relevant for structure design: Max. Value of total force over time
    Ftmax = np.max(Ft, axis=1)

    return Ftmax, zpm

if __name__ == "__main__":
    ftmax = calculate_maximum_forces()
    print(ftmax)

