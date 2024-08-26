import math
import time
import itertools
import sys
import json
import os
import io
import ast
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow_probability as tfp
#from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyDOE2 import lhs
from contextlib import redirect_stdout
from scipy.stats import linregress
from itertools import combinations_with_replacement


tf.keras.backend.set_floatx('float64')
record_frequency=100
#reproducibility
seed = 42
def reproducibility(seed = 42): 
    np.random.seed(seed)
    tf.random.set_seed(seed)

## Custom Layers
# Default parameters for the model
default_layers = [{'type': 'RFF', 'neuron': 2, 'mu': 0, 'sigma': 5, 'isTrainable': True},
                  {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
                  {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
                  {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
                  {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1}]

# custom layer: Random Fourier Features
class RFFLayer(tf.keras.layers.Layer):
    def __init__(self, units, mu=0, sigma=0.1, isTrainable=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.mu=mu
        self.sigma=sigma
        self.isTrainable=isTrainable

    def build(self, input_shape):
        # Initialize B with a Gaussian distribution
        self.B = self.add_weight(name="B", shape=(input_shape[-1], self.units),
                                 initializer=tf.random_normal_initializer(mean=self.mu, stddev=self.sigma),
                                 trainable=self.isTrainable)

    def call(self, inputs):
        # Perform the matrix multiplication and apply sin and cos
        xB = tf.matmul(inputs, self.B)
        return tf.concat([tf.sin(xB), tf.cos(xB)], axis=-1)

# custom layer: Polynomial Embedding
class PolynomialLayer(tf.keras.layers.Layer):   
    def __init__(self, degree, mu=0, sigma=0.1, isTrainable=False, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.mu=mu
        self.sigma=sigma
        self.isTrainable=isTrainable

    def build(self, input_shape):
        n_features = input_shape[1]
        self.terms = []
        for d in range(1, self.degree + 1):
            self.terms.extend(combinations_with_replacement(range(n_features), d))
        self.n_terms = len(self.terms)
        self.poly_weights = self.add_weight(shape=(self.n_terms,),
                                            initializer=tf.random_normal_initializer(mean=self.mu, stddev=self.sigma),
                                            trainable=self.isTrainable)

    def call(self, inputs):
        polynomial_terms = []
        for term in self.terms:
            product = tf.ones_like(inputs[:, 0])
            for idx in term:
                product *= inputs[:, idx]
            polynomial_terms.append(product)
        output = tf.stack(polynomial_terms, axis=1)
        return output * self.poly_weights
    
# custom layer: Random Weight Factorized
class RWFLayer(tf.keras.layers.Layer):
    def __init__(self, units, kernel_initializer='glorot_normal', activation='tanh', mu=1.0, sigma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.initializer = tf.keras.initializers.get(kernel_initializer)
        self.activation = tf.keras.activations.get(activation)
        self.mu=mu
        self.sigma=sigma

    def build(self, input_shape):
        # Initial weight matrix
        initial_weights = self.initializer(shape=(input_shape[-1], self.units))
        # Initialize scaling factors s
        self.s = self.add_weight(name='s_factors', shape=(input_shape[-1],),
                                 initializer=tf.random_normal_initializer(mean=self.mu, stddev=self.sigma),
                                 trainable=True)
        # Compute diag matrix of exp(s) and its inverse
        exp_s = tf.exp(self.s)
        diag_s_inv = tf.linalg.diag(1.0 / exp_s)
        # Compute V
        self.V = self.add_weight(name='V_matrix', shape=(input_shape[-1], self.units),
                                 initializer=lambda shape, dtype: tf.matmul(diag_s_inv, initial_weights),
                                 trainable=True)
        # Bias
        self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")
        super().build(input_shape)

    def call(self, inputs):
        # Compute W = diag(exp(s)) . V
        W = tf.matmul(tf.linalg.diag(tf.exp(self.s)), self.V)
        output = tf.matmul(inputs, W) + self.bias
        return self.activation(output)

## PINN Builder
def PINN_builder(in_shape=2, out_shape=1, layers=default_layers):
    tf.keras.backend.clear_session()
    model_name='PINN_'
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    x = input_layer
    for lyr in layers:
        if lyr['type']=='RFF':
            x = RFFLayer(lyr['neuron'], mu=lyr['mu'], sigma=lyr['sigma'], isTrainable=lyr['isTrainable'])(x)
            model_name+=f"RFF_{lyr['neuron']}_{lyr['mu']}_{lyr['sigma']}_{lyr['isTrainable']}_"
        elif lyr['type']=='Polynomial':
            x = PolynomialLayer(lyr['neuron'], mu=lyr['mu'], sigma=lyr['sigma'], isTrainable=lyr['isTrainable'])(x)
            model_name+=f"Polynomial_{lyr['neuron']}_{lyr['mu']}_{lyr['sigma']}_{lyr['isTrainable']}_"
        elif lyr['type']=='RWF':
            x = RWFLayer(lyr['neuron'], activation=lyr['act'], kernel_initializer=lyr['k_init'],
                         mu=lyr['mu'], sigma=lyr['sigma'])(x)
            model_name+=f"RWF_{lyr['neuron']}_{lyr['act']}_{lyr['k_init']}_{lyr['mu']}_{lyr['sigma']}_"
        else:
            x = Dense(lyr['neuron'], activation=lyr['act'], kernel_initializer=lyr['k_init'])(x)
            model_name += f"Dense_{lyr['neuron']}_{lyr['act']}_{lyr['k_init']}_"
    output_layer = Dense(out_shape,name='output_layer')(x)
    # building the model
    model = tf.keras.Model(input_layer, output_layer, name=model_name[:-1])
    return model

## Custom Loss
@tf.function
def custom_loss(u_pinn, collo_pts_batch, collo_subds_size_batch, diffEq_residu, eps, bds_pts_batch, bds_vals_batch):
    # Collocation loss
    phys_loss = tf.stack([tf.reduce_mean(tf.square(diffEq_residu(u_pinn, subd))) 
                          for subd in tf.split(collo_pts_batch, collo_subds_size_batch)])
    phy_weights= tf.exp(-eps*(tf.cumsum(phys_loss)-phys_loss))
    phys_loss=tf.reduce_mean(phy_weights*phys_loss)
    #phys_loss=diffEq_residu(u_pinn, collo_pts_batch)
    # Data loss
    vals_pred_batch = u_pinn(bds_pts_batch)
    bds_loss=tf.reduce_mean(tf.square(vals_pred_batch-bds_vals_batch))
    return (phys_loss, bds_loss)

## Custom Training Loop
def training_loop(loop_params):
    # Folder to save results
    dir=loop_params['dir']+'saved_model/'
    # PINN model
    u_pinn = loop_params['model']
    # Optimizer
    if 'opt' in loop_params:
        opt = loop_params['opt']
    else:
        lr_schedule = ExponentialDecay(loop_params['lr']['init_lr'],
                                       decay_steps=loop_params['lr']['decay_steps'],
                                       decay_rate=loop_params['lr']['decay_rate'], staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # Weights of data and physical loss
    weights=loop_params['phys_bds_w']['weights']
    update_steps=loop_params['phys_bds_w']['update_steps']
    moving_average_coef=loop_params['phys_bds_w']['moving_average_coef']
    # checkpoint
    best_loss = float('inf')
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=u_pinn)
    manager = tf.train.CheckpointManager(checkpoint, dir, max_to_keep=1)
    # Record results
    vals_loss = []
    num_batches=loop_params['num_batches']
    sp_boundaries=loop_params['data']['sp_boundaries']
    tr_collo_pts, tr_coeff_subds, tr_bds_pts, tr_bds_vals=loop_params['data']['training']
    dim=len(sp_boundaries)
    lower_sp_bds=np.array([bd['min'] for bd in sp_boundaries])
    upper_sp_bds=np.array([bd['max'] for bd in sp_boundaries])
    num_pts_bds=[0]
    for bd in (sp_boundaries):
        if 'fct_lower' in bd:num_pts_bds+=[bd['num_pts_bd']]
        if 'fct_upper' in bd:num_pts_bds+=[bd['num_pts_bd']]
    num_pts_bds=np.array(num_pts_bds)
    _, counts = np.unique(np.round(tr_coeff_subds, 2), return_counts=True)
    num_subds = sum(counts)
    size_subd = tr_collo_pts.shape[0] // num_subds
    # Batch size of collocation pts by subdomains
    size_subd_batch = size_subd // num_batches
    collo_subds_size_batch = tf.constant([count * size_subd_batch for count in counts], dtype=tf.int64)
    # Batch size of boundaries pts
    idx_bds = np.cumsum(num_pts_bds)
    bds_size_batch = tf.convert_to_tensor(np.diff(idx_bds) // num_batches, dtype=tf.int32)
    bds_ranges = [tf.range(idx_bds[i], idx_bds[i+1]) for i in range(len(idx_bds)-1)]
    num_epochs=loop_params['num_epochs']
    eps=loop_params['eps']
    diffEq_residu=loop_params['diffEq_residu']
    # Start training
    start = time.time()
    for epoch in range(num_epochs):
        # Batch for collocation Pts
        collo_ids_batch = tf.concat([tf.random.shuffle(tf.range(start, start + size_subd))[:size_subd_batch]
                                     for start in range(0, num_subds * size_subd, size_subd)], axis=0)
        collo_pts_batch = tf.gather(tr_collo_pts, collo_ids_batch)
        # Batch for Boundary Pts
        bds_ids_batch = tf.concat([tf.random.shuffle(rng)[:size]
                                   for rng, size in zip(bds_ranges, bds_size_batch)], axis=0)
        bds_pts_batch = tf.gather(tr_bds_pts, bds_ids_batch)
        bds_vals_batch = tf.gather(tr_bds_vals, bds_ids_batch)
        # Compute gradient
        with tf.GradientTape(persistent=True) as tape:
            arr_loss=custom_loss(u_pinn, collo_pts_batch, collo_subds_size_batch, diffEq_residu, eps, bds_pts_batch, bds_vals_batch)
            total_loss = tf.reduce_sum(weights*arr_loss)
        gradients = tape.gradient(total_loss, u_pinn.trainable_variables)
        opt.apply_gradients(zip(gradients, u_pinn.trainable_variables))
        # Update the weight
        if epoch % update_steps == 0 and epoch>0:
            gradient_norms = []
            for loss_value in arr_loss:
                gradients = tape.gradient(loss_value, u_pinn.trainable_variables)
                f_gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
                gradient_norms.append(float(tf.norm(f_gradients)))
            grads_loss = tf.constant(gradient_norms, dtype=tf.float64)
            grads_loss_sum=tf.reduce_sum(grads_loss)
            w_grads_loss=grads_loss/grads_loss_sum
            weights=moving_average_coef*weights+(1-moving_average_coef)*w_grads_loss
        del tape
        # Save the model
        if tf.reduce_sum(arr_loss) < best_loss:
            best_loss = tf.reduce_sum(arr_loss)
            manager.save()
        # Record the Loss
        if epoch % record_frequency == 0:
            vals_loss.append(arr_loss)
        if epoch % 1000 == 0:
            print(f"epoch: {epoch}, Loss: {arr_loss}", flush=True)
    # Restore best model
    checkpoint.restore(manager.latest_checkpoint)
    # End training
    end = time.time()
    tr_time=end-start
    return vals_loss, tr_time, u_pinn

## Generation Collocation and Boundaries Points
def sort_subd(coeffs, pts):
    # Step 1: Get the sorted indices of the elements in coeffs
    sorted_indices = np.argsort(coeffs.flatten())
    # Step 2: Initialize an empty array for sorted subarrays
    sorted_pts = np.empty_like(pts)
    # Step 4: Use sorted indices to sort subarrays in pts
    flat_pts = pts.reshape(-1, *pts.shape[len(coeffs.shape):])
    # Step 4: Use sorted indices to sort subarrays in pts
    for i, idx in enumerate(sorted_indices):
        sorted_pts[np.unravel_index(i, coeffs.shape)] = flat_pts[idx]
    return np.sort(coeffs.flatten()), sorted_pts

def reduce_coeffs(arr, n):
    flat_arr = arr.flatten()
    unique_values = np.unique(flat_arr)
    unique_values.sort()
    step = max(1, len(unique_values) // n)
    groups = [unique_values[i:i + step] for i in range(0, len(unique_values), step)]
    if len(groups) > n:
        groups[-2] = np.concatenate((groups[-2], groups[-1]))
        groups = groups[:-1]
    group_averages = [np.mean(group) for group in groups]
    # Create a mapping from original value to its group's average
    value_to_avg = {}
    for group, avg in zip(groups, group_averages):
        for value in group:
            value_to_avg[value] = avg
    reduced_flat_arr = np.vectorize(value_to_avg.get)(flat_arr)
    reduced_flat_arr = reduced_flat_arr/np.max(reduced_flat_arr)
    reduced_arr = reduced_flat_arr.reshape(arr.shape)
    return reduced_arr

def generate_pts(sp_boundaries, num_colloc_pts, num_subd):
    dim=len(sp_boundaries)
    lower_sp_bds=np.array([bd['min'] for bd in sp_boundaries])
    upper_sp_bds=np.array([bd['max'] for bd in sp_boundaries])
    len_sp_bds=upper_sp_bds-lower_sp_bds
    num_subd_sp_bds=np.array([bd['num_subdomains'] for bd in sp_boundaries])
    len_sp_subds=len_sp_bds/num_subd_sp_bds
    sp_subds = [np.linspace(lower_sp_bds[i], upper_sp_bds[i], num_subd_sp_bds[i]+1) for i in range(dim)]
    # For space collocation points
    num_pts_subd=int(num_colloc_pts/np.prod(num_subd_sp_bds))
    collo_pts = np.empty(tuple(num_subd_sp_bds) + (num_pts_subd, dim), dtype=object)
    for idx in np.ndindex(*num_subd_sp_bds):
        min_vals = [sp_subds[i][idx[i]] for i in range(dim)]
        collo_pts[idx]= lhs(dim, samples=num_pts_subd, random_state=seed)*len_sp_subds+min_vals
    # For coefficient
    segment_centers = [(edges[:-1] + edges[1:]) / 2 for edges in sp_subds]
    # Create meshgrid for all combinations of segment centers
    grids = np.meshgrid(*segment_centers, indexing='ij')
    # Flatten the grids to list all subdomain centers
    subdomain_centers = np.stack(grids, axis=-1).reshape(-1, dim)
    # Calculate the distance of each subdomain center from the hyperrectangle center
    distances = np.linalg.norm(subdomain_centers - (lower_sp_bds + upper_sp_bds)/2, axis=1)
    # Normalize distances
    normalized_distances = distances / np.linalg.norm((upper_sp_bds - lower_sp_bds)/2)
    # Define coefficients and Reshape
    subds_coefs = np.exp(-normalized_distances).reshape(num_subd_sp_bds)
    subds_coefs = reduce_coeffs(subds_coefs, num_subd)
    # sort collocation points
    subds_coefs, collo_pts=sort_subd(subds_coefs, collo_pts)
    collo_pts = collo_pts.reshape(np.prod(collo_pts.shape[:-1]), collo_pts.shape[-1])
    subds_coefs = subds_coefs.flatten()
    # Point for boundaries
    num_bds_pts=np.array([bd['num_pts_bd'] for bd in sp_boundaries])
    bds_pts=[]
    bds_vals=[]
    # Loop over each dimension to generate pts in boundaries
    for i in range(dim):
        size=num_bds_pts[i]
        if 'fct_lower' in sp_boundaries[i]:
            min_pts=np.insert(lhs(dim-1, samples=size, random_state=seed)*np.delete(len_sp_bds, i) + np.delete(lower_sp_bds, i),
                          i, [lower_sp_bds[i]]*size, axis=1)
            bds_pts.extend(min_pts)
            bds_vals.extend([sp_boundaries[i]['fct_lower'](pt) for pt in np.delete(min_pts, i, axis=1)])
        if 'fct_upper' in sp_boundaries[i]:
            max_pts=np.insert(lhs(dim-1, samples=size, random_state=seed)*np.delete(len_sp_bds, i) + np.delete(lower_sp_bds, i),
                          i, [upper_sp_bds[i]]*size, axis=1)
            bds_pts.extend(max_pts)
            bds_vals.extend([sp_boundaries[i]['fct_upper'](pt) for pt in np.delete(max_pts, i, axis=1)])
    bds_vals=np.array(bds_vals)
    return tf.convert_to_tensor(collo_pts, dtype=tf.float64),\
           tf.convert_to_tensor(subds_coefs, dtype=tf.float64),\
           tf.convert_to_tensor(bds_pts, dtype=tf.float64), tf.convert_to_tensor(bds_vals, dtype=tf.float64)

## Plot Collocation and Boundaries Points
def plot_collo_bds_pts(collo_pts, coeff_subds, bds_pts):
    cmap = plt.get_cmap('rainbow')
    plt.figure(figsize=(12, 8))
    plt.title("Boundary Points & Collocation Points",fontsize=18)
    _, counts = np.unique(np.round(coeff_subds, 2), return_counts=True)
    size_subd=collo_pts.shape[0]//sum(counts)
    cumul=0
    for i in range(len(counts)):
        plt.scatter(collo_pts[cumul:cumul+counts[i]*size_subd, 0],
                    collo_pts[cumul:cumul+counts[i]*size_subd, 1], s=2, marker='.',
                    color=cmap(i / len(counts)), label='CP')
        cumul+=counts[i]*size_subd
    plt.scatter(bds_pts[:, 0], bds_pts[:, 1], s=1, marker='+', c='gray', label='BDP')
    plt.xlabel("Time axis t", fontsize=16, labelpad=8)
    plt.ylabel("Space axis x", fontsize=16, labelpad=8)
    return plt

## Plot Loss
def plot_loss(vals_loss, loop_params):
    lst_epo=10000//record_frequency
    epochs = np.arange(0, loop_params['num_epochs'], record_frequency)
    last_epochs = epochs[-lst_epo:]
    vals_loss=np.array(vals_loss)
    last_vals_loss = vals_loss[-lst_epo:]
    plt.figure(figsize=(12, 8))
    
    red =  np.array([1, 0, 0])
    lightred = .4*red + .6*np.array([1, 1, 1])
    plt.plot(epochs, vals_loss[:, 0], label='Physical Loss',  linestyle='-', color=lightred)
    slope_phys, intercept, _,_,_= linregress(last_epochs, np.log(last_vals_loss[:, 0]))
    line = np.exp(intercept) * np.exp(slope_phys * last_epochs)
    plt.plot(last_epochs, line, color=red)
    plt.text(last_epochs[-1], line[-1], f"{slope_phys:.0e}", verticalalignment='bottom', horizontalalignment='right', color=red, fontsize=13)
    
    green =  np.array([0, 1, 0])
    lightgreen = .4*green + .6*np.array([1, 1, 1])
    plt.plot(epochs, vals_loss[:, 1], label='Boundaries Loss',  linestyle='-', color=lightgreen)
    slope_data, intercept, _,_,_= linregress(last_epochs, np.log(last_vals_loss[:, 1]))
    line = np.exp(intercept) * np.exp(slope_data * last_epochs)
    plt.plot(last_epochs, line, color=green)
    plt.text(last_epochs[-1], line[-1], f"{slope_data:.0e}", verticalalignment='bottom', horizontalalignment='right', color=green, fontsize=13)
    
    blue =  np.array([0, 0, 1])
    lightblue = .4*blue + .6*np.array([1, 1, 1])
    total_loss=vals_loss[:, 0]+vals_loss[:, 1]
    plt.plot(epochs, total_loss, label='Total Loss', linestyle='-', color=lightblue)
    slope_total, intercept, _,_,_= linregress(last_epochs, np.log(total_loss[-lst_epo:]))
    line = np.exp(intercept) * np.exp(slope_total * last_epochs)
    plt.plot(last_epochs, line, color=blue)
    plt.text(last_epochs[-1], line[-1], f"{slope_total:.0e}", verticalalignment='bottom', horizontalalignment='right', color=blue, fontsize=13)

    plt.yscale('log')
    plt.xlabel(f'Epoch 10^3', fontsize=16, labelpad=8)
    plt.ylabel('Loss', fontsize=16, labelpad=8)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x // 1000:.0f}'))
    plt.title('Progress of the loss',fontsize=18)
    plt.legend()
    return plt, slope_phys, slope_data, slope_total

## Plot the predicted u
def plot_model(u_pinn, sp_boundaries):
    t_vals = np.linspace(sp_boundaries[0]['min'], sp_boundaries[0]['max'], 100)
    x_vals = np.linspace(sp_boundaries[1]['min'], sp_boundaries[1]['max'], 100)
    t_grid, x_grid = np.meshgrid(t_vals, x_vals)
    inputs = np.vstack([t_grid.ravel(), x_grid.ravel()]).T
    u_pred = u_pinn.predict(inputs, verbose=0).reshape(t_grid.shape)

    # 3D Plot
    fig1 = plt.figure(figsize=(12, 8))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_surface(t_grid, x_grid, u_pred, cmap='jet')
    ax.set_xlabel('Time axis t', fontsize=16, labelpad=8)
    ax.set_ylabel('Space axis x', fontsize=16, labelpad=8)
    ax.set_zlabel('u(t,x)', fontsize=16, labelpad=8)
    ax.view_init(20, 310)
    # Heatmap
    fig2 = plt.figure(figsize=(12, 8))
    plt.pcolor(t_grid, x_grid, u_pred, shading='auto')
    plt.colorbar()
    plt.title("u(t,x)", fontsize=16)
    plt.xlabel("Time axis t", fontsize=16, labelpad=8)
    plt.ylabel("Space axis x", fontsize=16, labelpad=8)
    return fig1, fig2
    
## Plot Absolut Error
def plot_error(u_pinn, sp_boundaries):
    t_vals = np.linspace(sp_boundaries[0]['min'], sp_boundaries[0]['max'], 100)
    x_vals = np.linspace(sp_boundaries[1]['min'], sp_boundaries[1]['max'], 100)
    t_grid, x_grid = np.meshgrid(t_vals, x_vals)
    err_vals = np.zeros((len(x_vals), len(t_vals)))
    for t_id, x_id in itertools.product(range(len(t_vals)),range(len(x_vals))):
        t, x = t_vals[t_id], x_vals[x_id]
        if t==sp_boundaries[0]['min'] and 'fct_lower' in sp_boundaries[0]:
            err_vals[x_id, t_id] =u_pinn(tf.constant([[t, x]], dtype=tf.float64))[0,0].numpy()- sp_boundaries[0]['fct_lower'](x)
        elif t==sp_boundaries[0]['max'] and 'fct_upper' in sp_boundaries[0]:
            err_vals[x_id, t_id] =u_pinn(tf.constant([[t, x]], dtype=tf.float64))[0,0].numpy()- sp_boundaries[0]['fct_upper'](x)
        elif x==sp_boundaries[1]['min'] and 'fct_lower' in sp_boundaries[1]:
            err_vals[x_id, t_id] =u_pinn(tf.constant([[t, x]], dtype=tf.float64))[0,0].numpy()- sp_boundaries[1]['fct_lower'](x)
        elif x==sp_boundaries[1]['max'] and 'fct_upper' in sp_boundaries[1]:
            err_vals[x_id, t_id] =u_pinn(tf.constant([[t, x]], dtype=tf.float64))[0,0].numpy()-sp_boundaries[1]['fct_upper'](x)        
        else:
            err_vals[x_id, t_id] = heatEq_residu(u_pinn, tf.constant([[t, x]], dtype=tf.float64))[0,0].numpy()
    plt.figure(figsize=(12, 8))
    plt.pcolor(t_grid, x_grid, err_vals, shading='auto')    
    plt.colorbar()
    plt.title("Error",fontsize=18)
    plt.xlabel("Time axis t", fontsize=16, labelpad=8)
    plt.ylabel("Space axis x", fontsize=16, labelpad=8)
    return plt

## Heat Equation
dir='model_test/'
time_t = {
    'min': 0,
    'max': 10,
    'num_subdomains': 7,
    'num_pts_bd': 1000,
    'fct_lower': lambda x: np.sin(2 * x)
}
sp_x = {
    'min': 0,
    'max': 1,
    'num_subdomains': 1,
    'num_pts_bd': 100,
    'fct_lower': lambda t: np.cos(np.pi*t),
    'fct_upper': lambda t: np.sin(np.pi*t)
}
@tf.function
def heatEq_residu(u_pinn, vect):
    alpha=1
    t, x = tf.split(vect, num_or_size_splits=2, axis=1)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([t, x])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([t, x])
            u0 =  u_pinn(tf.concat([t, x], axis=1))
        u_t = tape1.gradient(u0, t)
        u_x = tape1.gradient(u0, x)
    u_xx = tape2.gradient(u_x, x)
    del tape1
    del tape2
    return u_t -alpha*u_xx-tf.exp(-t)*tf.sin(3*x)

layers = [{'type': 'RFF', 'neuron': 2, 'mu': 0, 'sigma': 1, 'isTrainable': True},
          {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
          {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
          {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
          {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1},
          {'type': 'RWF', 'neuron': 50, 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0, 'sigma': 0.1}]

phys_bds_w = {'weights': tf.constant([1, 1], dtype=tf.float64), 
              'update_steps': 1000, 
              'moving_average_coef': 0.9}
lr = {'init_lr': 5e-4, 
      'decay_steps': 2000, 
      'decay_rate':0.9}
loop_params={'num_epochs': 60000, 
             'num_batches': 5, 
             'eps': 0.4,
             'diffEq_residu': heatEq_residu, 
             'phys_bds_w': phys_bds_w, 
             'lr': lr,
             'num_steps_LBFGSB': 5000}

if len(sys.argv)==2:
    arg_str = sys.argv[1]
    dict_arg = ast.literal_eval(arg_str)
    dir=os.path.splitext(sys.argv[0])[0]+'/model_'+str(dict_arg['id']) +'/'
    time_t['num_subdomains']=dict_arg['num_subdomains_eps'][0]
    ly_dense={'type': 'Dense', 'neuron': dict_arg['layers_neuron'], 'act': 'tanh', 'k_init': 'glorot_normal'}
    ly_RWF={'type': 'RWF', 'neuron': dict_arg['layers_neuron'], 'act': 'tanh', 'k_init': 'glorot_normal', 'mu': 1.0,'sigma': 0.1}
    #layers
    layers = [ly_dense] if dict_arg['HL_type']=='Dense' else [ly_RWF]
    layers=layers*dict_arg['layers_numb']
    if dict_arg['embedding']!='Without':
        ly_embedding={'type': dict_arg['embedding'], 'neuron': dict_arg['embedding_dim'], 'mu': 0, 
                      'sigma': 1, 'isTrainable': dict_arg['trainibility_embd'] == "True"}
        layers = [ly_embedding]+layers
    phys_bds_w['moving_average_coef']=dict_arg['moving_average_coef']
    loop_params['num_batches']= dict_arg['num_batches']
    loop_params['eps']=dict_arg['num_subdomains_eps'][1]

sp_boundaries=[time_t, sp_x]
dir='./Res/'+dir
loop_params['dir'] = dir
os.makedirs(dir, exist_ok=True)
# Check which device
with open(dir+'result.txt', 'a') as file:
    if tf.test.gpu_device_name():
        file.write('Default GPU Device: {}'.format(tf.test.gpu_device_name())+'\n')
    else:
        file.write("No GPU found, using CPU.\n")
# Generate Data
reproducibility()
tr_collo_pts, tr_coeff_subds, tr_bds_pts, tr_bds_vals=generate_pts(sp_boundaries, 5000, 40)
vl_collo_pts, vl_coeff_subds, vl_bds_pts, vl_bds_vals=generate_pts(sp_boundaries, 5000, 40)
#Plot data points
plt=plot_collo_bds_pts(tr_collo_pts, tr_coeff_subds, tr_bds_pts)
plt.savefig(dir+"plot_collo_bds_pts.png")
plt.close()
data={'sp_boundaries':sp_boundaries,
      'training': (tr_collo_pts, tr_coeff_subds, tr_bds_pts, tr_bds_vals)}
loop_params['data'] = data

# Build a Model
u_pinn=PINN_builder(2, 1, layers)
loop_params['model'] = u_pinn
with open(dir+'result.txt', 'a') as file:
    f = io.StringIO()
    with redirect_stdout(f):
        u_pinn.summary()
    summary = f.getvalue()
    file.write(summary)

#Training
reproducibility(None)
vals_loss, tr_time, u_pinn = training_loop(loop_params)
vl_collo_subds_size=tf.constant([len(vl_collo_pts)], dtype=tf.int64)
validation_loss = custom_loss(u_pinn, vl_collo_pts, vl_collo_subds_size, heatEq_residu, 0, vl_bds_pts, vl_bds_vals)

#Save validation_loss
with open(dir+'result.txt', 'a') as file:
    file.write(f"Adam Training Time: {tr_time:.0f}\n")
    file.write(f"Adam Validation Physics Loss: {validation_loss[0] :.11f}\n")
    file.write(f"Adam Validation Data Loss: {validation_loss[1] :.11f}\n")
np.savetxt(dir+'vals_loss.txt', np.array(vals_loss))

#Plots
plt, slope_phys, slope_data, slope_total=plot_loss(vals_loss, loop_params)
plt.savefig(dir+"plot_loss.png")
plt.close()
with open(dir+'result.txt', 'a') as file:
    file.write(f"Slope Physics Loss: {slope_phys :.11f}\n")
    file.write(f"Slope Data Loss: {slope_data :.11f}\n")
    file.write(f"Slope Total Loss: {slope_total :.11f}\n")

plt1, plt2=plot_model(u_pinn, sp_boundaries)
plt1.savefig(dir+"plot_model_3d.png")
plt2.savefig(dir+"plot_model_HM.png")
plt=plot_error(u_pinn, sp_boundaries)
plt.savefig(dir+"plot_error_Adam.png")
plt.close()

# L-BFGS optimization step
def opt_LBFGSB(max_iterations=1000):
    initial_weights = tf.concat([tf.reshape(v, [-1]) for v in u_pinn.trainable_variables], axis=0)
    def set_weights(weights):
        idx = 0
        for v in u_pinn.trainable_variables:
            shape = v.shape
            size = tf.reduce_prod(shape)
            v.assign(tf.reshape(weights[idx:idx+size], shape))
            idx += size
                    
    def obj_func(weights):
        set_weights(weights)
        with tf.GradientTape() as tape:      
            arr_loss=custom_loss(u_pinn, tr_collo_pts, 
                                 tf.constant([len(tr_collo_pts)], dtype=tf.int64), 
                                 heatEq_residu, 0, tr_bds_pts, tr_bds_vals)
            total_loss = tf.reduce_sum(arr_loss)
        gradients = tape.gradient(total_loss, u_pinn.trainable_variables)
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
        return total_loss, flat_grads
    
    start = time.time()
    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=lambda w: obj_func(w),
                                           initial_position=initial_weights, max_iterations=max_iterations)
    end = time.time()
    opt_time=end-start
    set_weights(results.position)
    return opt_time, u_pinn

opt_time, u_pinn=opt_LBFGSB(loop_params['num_steps_LBFGSB'])
validation_loss = custom_loss(u_pinn, vl_collo_pts, vl_collo_subds_size, heatEq_residu, 0, vl_bds_pts, vl_bds_vals)
with open(dir+'result.txt', 'a') as file:
    file.write(f"LBFGSB Optimization Time: {opt_time:.0f}\n")
    file.write(f"LBFGSB Validation Physics Loss: {validation_loss[0] :.11f}\n")
    file.write(f"LBFGSB Validation Data Loss: {validation_loss[1] :.11f}\n")

plt=plot_error(u_pinn, sp_boundaries)
plt.savefig(dir+"plot_error_LBFGSB.png")
plt.close()
u_pinn.save(dir + 'final_model.keras')