import os
import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from concurrent.futures import ProcessPoolExecutor

pi = np.pi
precis=1e-10
n_terms=1000

## Heat Equation with Dirichlet BCs
def analytical_sol_D(f, g, pt):
    def g_term(n):
        return 2*quad(lambda x: g(x)*np.sin(n*pi*x), 0, 1, epsabs=precis, epsrel=precis)[0]
    def f_term(n, t):
        def integrand_f_term(x, s, n, t): return np.exp(-(t-s)*(n*pi)**2)*np.sin(n*pi*x)*f(t,x)
        return 2*dblquad(integrand_f_term, 0, t, lambda s: 0, lambda s: 1, args=(n, t), epsabs=precis, epsrel=precis)[0]
    t,x=pt[0],pt[1]
    exp_t_n = np.exp(-t*(np.arange(1, n_terms+1)*pi)**2)
    g_terms = np.array([g_term(n) for n in range(1, n_terms+1)])
    f_terms=np.array([f_term(n, t) for n in range(1, n_terms+1)])
    sum_t_n=exp_t_n*g_terms+f_terms
    sin_x_n = np.sin(x*np.arange(1, n_terms+1)*pi)
    return np.sum(sum_t_n*sin_x_n)

## Heat Equation with Neumman BCs
def analytical_sol_N(f, g, pt): 
    t,x=pt[0],pt[1]
    def z(pt):
        integral_g = quad(g, 0, 1, epsabs=precis, epsrel=precis)[0]
        sum_terms = 0
        for n in range(1, n_terms + 1):
            cos_term = quad(lambda x: g(x)*np.cos(n*np.pi*x), 0, 1, epsabs=precis, epsrel=precis)[0]
            sum_terms += np.exp(-t*(n * np.pi)**2) * np.cos(n*np.pi*x)*cos_term
        return integral_g + 2 * sum_terms
    def w(pt):
        t,x=pt[0],pt[1]
        integral_F = dblquad(lambda x, s: f(x, s), 0, t, lambda x: 0, lambda x: 1, epsabs=precis, epsrel=precis)[0]
        sum_terms = 0
        for n in range(1, n_terms + 1):
            inner_integral = dblquad(lambda x, s: np.exp(-(n * np.pi)**2*(t - s))*f(x, s)*np.cos(n*np.pi* x), 0, t, lambda x: 0, lambda x: 1, epsabs=precis, epsrel=precis)[0]
            sum_terms += np.cos(n * np.pi * x) * inner_integral
        return integral_F + 2 * sum_terms
    return z(pt) + w(pt)

#Grid
t_vals = np.linspace(0, 10, 1000)
x_vals = np.linspace(0, 1, 100)
t_grid, x_grid = np.meshgrid(t_vals, x_vals)
inputs = np.vstack([t_grid.ravel(), x_grid.ravel()]).T

# Get the number of CPUs
num_cpus = os.cpu_count()
print(f"Number of CPUs available: {num_cpus}", flush=True)


# Discontinuous
## Dirichlet
### Eq1
def f(t,x): return np.exp(-t)*np.sin(3*x)+pi*(1-x)*np.sin(pi*t)-pi*x*np.cos(pi*t)
def g(x): return np.sin(2 * x)+x-1
def H(t,x): return (np.sin(pi*t)-np.cos(pi*t))*x+np.cos(pi*t)
# v solution
def compute_analytical_D(pt): return analytical_sol_D(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_D, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Discon_D_1_v'+'.npy', v_vals)
# u solution
u_vals= v_vals + H(inputs[:,:1], inputs[:,1:]).reshape(t_grid.shape)
np.save('Discon_D_1_u'+'.npy', u_vals)

### Eq2
def f(t,x): return np.exp(x**2-t)+2*(1-x)*np.exp(-2*t)-x*(1-t)*np.exp(-t)
def g(x): return np.exp(-x)+4*x-1 
def H(t,x): return (t*np.exp(-t)-np.exp(-2*t))*x+np.exp(-2*t)
# v solution
def compute_analytical_D(pt): return analytical_sol_D(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_D, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Discon_D_2_v'+'.npy', v_vals)
# u solution
u_vals= v_vals + H(inputs[:,:1], inputs[:,1:]).reshape(t_grid.shape)
np.save('Discon_D_2_u'+'.npy', u_vals)

## Neumann
### Eq1
def f(t,x): return np.exp(-t)*np.sin(3*x)-np.cos(pi*t)+np.sin(pi*t)+((pi*x)/2)*((2-x)*np.sin(pi*t)-x*np.cos(pi*t))
def g(x): return np.sin(2 * x)+((x**2)/2)-x
def H(t,x): return (x/2)*((2-x)*np.cos(pi*t)+x*np.sin(pi*t))
# v solution
def compute_analytical_N(pt): return analytical_sol_N(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_N, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Discon_N_1_v'+'.npy', v_vals)
# u solution
u_vals= v_vals + H(inputs[:,:1], inputs[:,1:]).reshape(t_grid.shape)
np.save('Discon_N_1_u'+'.npy', u_vals)

### Eq2
def f(t,x): return np.exp(x**2-t)-np.exp(-2*t)+t*np.exp(-t)+(x/2)*np.exp(-t)*(2*(2-x)*np.exp(-t)-x*(1-t))
def g(x): return np.exp(-x)+((x**2)/2)+2*x
def H(t,x): return (x/2)*((2-x)*np.exp(-2*t)+x*t*np.exp(-t))
# v solution
def compute_analytical_N(pt): return analytical_sol_N(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_N, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Discon_N_2_v'+'.npy', v_vals)
# u solution
u_vals= v_vals + H(inputs[:,:1], inputs[:,1:]).reshape(t_grid.shape)
np.save('Discon_N_2_u'+'.npy', u_vals)

# Continuous
## v Dirichlet
### Eq1
def f(t,x): return np.sin(3*x)
def g(x): return np.sin(np.pi*x)*np.cos(np.pi*x)
# v solution
def compute_analytical_D(pt): return analytical_sol_D(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_D, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Con_D_1_v'+'.npy', v_vals)

### Eq2
def f(t,x): return np.exp(x**2-t)
def g(x): return (1-x)*x**2
# v solution
def compute_analytical_D(pt): return analytical_sol_D(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_D, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Con_D_2_v'+'.npy', v_vals)

## v Neumann
### Eq1
def f(t,x): return np.sin(3*x)
def g(x): return np.sin(np.pi*x)*np.cos(np.pi*x)
# v solution
def compute_analytical_N(pt): return analytical_sol_N(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_N, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Con_N_1_v'+'.npy', v_vals)

### Eq2
def f(t,x): return np.exp(x**2-t)
def g(x): return (1-x)*x**2
# v solution
def compute_analytical_N(pt): return analytical_sol_N(f, g, pt)
with ProcessPoolExecutor() as executor:
    v_vals = list(executor.map(compute_analytical_N, inputs))
v_vals= np.array(v_vals).reshape(t_grid.shape)
np.save('Con_N_2_v'+'.npy', v_vals)