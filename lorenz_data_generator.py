import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def generate_lorenz_data(params=None):
    """
    Generate data for the Lorenz system using time integration
    
    Args:
        params (dict): Dictionary containing parameters:
            sigma (float): First parameter of the Lorenz system
            rho (float): Second parameter of the Lorenz system
            beta (float): Third parameter of the Lorenz system
            dt (float): Time step
            T (float): Total simulation time
            num_steps (int): Number of time steps (optional, if not provided will use dt)
    
    Returns:
        tuple: (t, xyz) where xyz contains the solution values for x, y, z coordinates
    """
    if params is None:
        raise ValueError("Parameters must be provided")
    
    sigma, rho, beta = params['sigma'], params['rho'], params['beta']
    dt, T = params['dt'], params['T']
    
    def lorenz_rhs(t, state):
        """Right hand side of the Lorenz system"""
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]
    
    # Initial condition
    xyz0 = [1.0, 1.0, 1.0]
    
    # Time points
    if 'num_steps' in params:
        t = np.linspace(0, T, params['num_steps'])
    else:
        t = np.arange(0, T + dt, dt)
    
    # Solve the system
    sol = solve_ivp(
        lorenz_rhs,
        [0, T],
        xyz0,
        t_eval=t,
        method='RK45',
        rtol=1e-6,
        atol=1e-6,
    )
    
    # Get solution
    t, xyz = sol.t, sol.y.T
    
    return t, xyz

def plot_solution(t, xyz):
    """Plot the Lorenz attractor solution"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz Attractor')
    plt.savefig("lorenz_solution.png", dpi=300)
    plt.show(block=False)

if __name__ == "__main__":
    import os

    # Set up parameters for Lorenz system
    params = {
        'sigma': 1,
        'rho': 2,
        'beta': 3,
        'dt': 0.01,
        'T': 100,
        'num_steps': 10001  # Total steps for 0 to 100
    }
    # Generate data
    t, xyz = generate_lorenz_data(params)
    
    # Plot the solution
    plot_solution(t, xyz)
    
    # Save in the required format and location
    DATA_FOLDER = "data"
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Split data into training (0-5000) and truth (5001-10001)
    training_data = xyz[0:5001]  # First 5001 time steps (0 to 50)
    truth_data = xyz[5001:10001]  # Last 5000 time steps (50 to 100)
    
    # Save the training and truth data
    TRAINING_FILE = os.path.join(DATA_FOLDER, "lorenz_training2.npy")
    TRUTH_FILE = os.path.join(DATA_FOLDER, "lorenz_truth2.npy")
    
    np.save(TRAINING_FILE, training_data)
    np.save(TRUTH_FILE, truth_data)
    
    print(f"Generated data files in: {DATA_FOLDER}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Truth data shape: {truth_data.shape}")
    print(f"T range: [{t.min():.2f}, {t.max():.2f}]")
