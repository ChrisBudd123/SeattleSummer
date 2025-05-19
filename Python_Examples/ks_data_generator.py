import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def generate_ks_data(params=None):
    """
    Generate data for the Kuramoto-Sivashinsky equation using time integration
    
    Args:
        params (dict): Dictionary containing parameters:
            L (float): Domain length
            N (int): Number of spatial points
            nu (float): Viscosity parameter
            dt (float): Time step
            T (float): Total simulation time
            num_steps (int): Number of time steps (optional, if not provided will use dt)
    
    Returns:
        tuple: (x, t, u) where u contains the solution values
    """
    if params is None:
        raise ValueError("Parameters must be provided")
    
    L, N, nu = params['L'], params['N'], params['nu']
    dt, T = params['dt'], params['T']
    
    # Spatial domain
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N
    
    # Wavenumbers for spectral method
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    
    # Initial condition
    u0 = 1./(2+np.cos(x))
    
    def KS_rhs(t, u):
        """Right hand side of the KS equation"""
        u_hat = np.fft.fft(u)
        u_x = np.fft.ifft(1j * k * u_hat).real
        u_xx = np.fft.ifft((1j * k)**2 * u_hat).real
        u_xxxx = np.fft.ifft((1j * k)**4 * u_hat).real
        return -u * u_x - nu * u_xx - nu * u_xxxx
    
    # Time points
    if 'num_steps' in params:
        t = np.linspace(0, T, params['num_steps'])
    else:
        t = np.arange(0, T + dt, dt)
    
    # Solve the equation
    sol = solve_ivp(
        KS_rhs,
        [0, T],
        u0,
        t_eval=t,
        method='BDF',
        rtol=1e-6,
        atol=1e-6,
    )
    
    # Reshape solution for output format
    t, u = sol.t, sol.y.T
    
    return x, t, u

def plot_solution(x, t, u):
    """Plot the KS solution"""
    plt.figure(figsize=(10, 6))
    X, T = np.meshgrid(x, t)
    plt.pcolormesh(X, T, u, shading='auto')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Kuramoto-Sivashinsky Solution')
    plt.savefig("ks_solution.png", dpi=300)
    plt.show(block=False)

if __name__ == "__main__":
    import os

    # Set up parameters for KS equation
    params = {
        'L': 30,
        'N': 2048,
        'nu': 0.5,
        'dt': 0.5,
        'T': 100.0,
        'num_steps': 201  # Total steps for 0 to 100
    }
    
    # Generate data
    x, t, u = generate_ks_data(params)
    
    # Plot the solution
    plot_solution(x, t, u)
    
    # Save in the required format and location
    DATA_FOLDER = "data"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Split data into training (0-100) and truth (101-201)
    training_data = u[0:int((params['num_steps']-1)/2)+1]  # First 101 time steps (0 to 100)
    truth_data = u[int((params['num_steps']-1)/2)+1:params['num_steps']]     # Last 101 time steps (100 to 200)
    
    # Save the training and truth data
    TRAINING_FILE = os.path.join(DATA_FOLDER, "ks_training2.npy")
    TRUTH_FILE = os.path.join(DATA_FOLDER, "ks_truth2.npy")
    
    np.save(TRAINING_FILE, training_data)
    np.save(TRUTH_FILE, truth_data)
    
    print(f"Generated data files in: {DATA_FOLDER}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Truth data shape: {truth_data.shape}")
    print(f"X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"T range: [{t.min():.2f}, {t.max():.2f}]")
