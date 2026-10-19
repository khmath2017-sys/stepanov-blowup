import numpy as np
from scipy.integrate import solve_ivp

def solve_heat_equation(Nx=200, dt=5e-4, t_max=0.6):
    """
    Solves the nonlinear heat equation u_t = u_xx + u^2 
    with Dirichlet boundary conditions u(0,t)=u(1,t)=0.
    Initial condition: u(x,0) = 3 sin(pi x).
    """
    x = np.linspace(0, 1, Nx)
    dx = x[1] - x[0]
    u0 = 3 * np.sin(np.pi * x)
    
    def rhs(t, u):
        # Finite difference for u_xx
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        # Dirichlet boundary conditions
        u_xx[0] = 0
        u_xx[-1] = 0
        # Reaction term u^2
        return u_xx + u**2

    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    # Solve ODE using an implicit method suitable for stiff equations near blow-up
    sol = solve_ivp(rhs, t_span, u0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)
    return sol.t, sol.y, x
