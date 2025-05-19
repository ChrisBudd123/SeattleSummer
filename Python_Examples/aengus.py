#imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

#global constants
EPSILON = .01

#PROBLEM:
# -d^2u / dx^2 = f
# u(-1) = tanh(-1)
# u(1) = tanh(1)

#define PINN neural network using nn.Module class
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

#neural network functions
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), 
create_graph=True)[0]

def forcing_function(x):
    return 2*torch.tanh(x)/(torch.cosh(x))**2

def compute_loss(model: nn.Module, x: torch.Tensor):
    #evaluating u(x), f(x), du/dx, d^2u/dx^2 at collocation points x
    x.requires_grad_(True)
    u = model(x)
    du = grad(u,x)
    d2u = grad(du,x)
    f = forcing_function(x)

    #ODE residual: -u"(x) - f(x)
    residual = -d2u - f
    interior_loss = torch.mean(residual ** 2)

    #calculating boundary loss for problem
    left_residual = model(torch.tensor([-1.0])) - torch.tanh(torch.tensor([-1.0]))
    right_residual = model(torch.tensor([1.0])) - torch.tanh(torch.tensor([1.0]))
    boundary_loss = torch.mean(left_residual ** 2 + right_residual ** 2)

    return interior_loss + boundary_loss


#Generating collocation points
def generate_collocation_points(method='uniform', num_points=10):
    if method == 'uniform':
        x_train = np.linspace(-1, 1, num_points)
    elif method == 'gauss_legendre':
        x_train = roots_legendre(num_points)[0]
    else:
        raise ValueError("Unsupported quadrature method")

    return torch.tensor(x_train.reshape(-1,1), dtype=torch.float32)

def train_PINN(x_train):
    # Training the PINN
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Iterate training over epochs
    for epoch in range(4000):
        loss = compute_loss(model, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model

# Helper function to plot results
def create_results(x_train, color='red', label=''):
    model = train_PINN(x_train)
    y_pred = model(x_train).detach().numpy()
    plt.plot(x_train.detach().numpy(), y_pred, label=label, color=color, linestyle='--')

if __name__ == "__main__":
    # Plotting True Result
    x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y_true = torch.tanh(x_test).numpy()
    plt.plot(x_test.numpy(), y_true, label= r'True Solution: $u^*(x) = \tanh(x)$', color='green')

    # Getting Collocation Points and weights
    uniform = generate_collocation_points(method='uniform')
    gauss_legendre = generate_collocation_points(method='gauss_legendre')

    #Plotting results
    create_results(uniform, color='blue',label='Uniform')
    create_results(gauss_legendre, color='green',label='Gauss-Legendre')

    #Plotting Prettiness
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-u''(x) = f(x)$"
    plt.title(title)
    plt.show()
