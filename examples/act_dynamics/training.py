import sys, os, time

sys.path += ["../../external/"]

import matplotlib.pyplot as plt
import torch
import gpytorch
import copy

from sklearn.cluster import KMeans
from pendulum_model import *
from utils import *

# gpytorch_utils
from gpytorch_utils.gp_hyperparam_training import (
    generate_train_inputs_acados,
    generate_train_outputs_at_inputs,
    train_gp_model,
)

from gpytorch import likelihoods
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentMultitaskGPModel,
    BatchIndependentInducingPointGpModel
)

dataset_path = "X_sim_with_gp_targets.csv"
n_gp_outputs = 2
train_inputs, train_outputs = load_gp_data_from_csv(dataset_path, n_gp_outputs)

#permuted_indices = torch.randperm(train_inputs.size(0))
#train_dim = int(0.70*train_inputs.size(0))
#x_train = train_inputs[permuted_indices[:train_dim], :]
#y_train = train_outputs[permuted_indices[:train_dim], :]
#x_val = train_inputs[permuted_indices[train_dim:], :]
#y_val = train_outputs[permuted_indices[train_dim:], :]
#
print(f"x_train: {train_inputs.shape}, y_train: {train_outputs.shape}")
#print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")

#kmeans = KMeans(n_clusters = 1000)
#kmeans.fit(train_inputs.numpy())
#_, indices = np.unique(kmeans.labels_, return_index=True)
#x_train = train_inputs[indices,:]
#y_train = train_outputs[indices,:]
#print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
##x_inducing_points = torch.tensor(kmeans.cluster_centers_, dtype = torch.float32)
##train_inputs = torch.tensor(train_inputs, dtype=torch.float32) 
##train_outputs = torch.tensor(train_outputs, dtype = torch.float32)
#train_x_y = np.hstack([x_train, y_train])
#np.savetxt("train_set.csv", train_x_y, delimiter=',', header='x1,x2,x3,x4,u,Y_p,Y_theta,Y_v,Y_w')

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = n_gp_outputs)

gp_model = BatchIndependentMultitaskGPModel(
    train_x = train_inputs,
    train_y = train_outputs,
    input_dimension=train_inputs.shape[1], # 4 + 1 = 5
    residual_dimension=n_gp_outputs,
    likelihood=likelihood,
    #use_ard=True,    # 
    #inducing_points = 100,
)

# Train the GP model on the data  
gp_model.train()
likelihood.train()
print("Training the GP model...")
gp_model, likelihood = train_gp_model(
    gp_model, training_iterations=300, learning_rate=0.05)#, val_x=x_val, val_y=y_val, compute_val_loss=True)
print("GP model trained successfully")
save_path = "gp_model.pth"
# Save state dicts and training data (optional if not used later)
torch.save({
    'model_state_dict': gp_model.state_dict(),
    'likelihood_state_dict': likelihood.state_dict(),
    'train_x': gp_model.train_inputs[0],  # if needed
    'train_y': gp_model.train_targets,    # if needed
}, save_path)

# Plot the GP model fit on training data
#plot_gp_fit_on_training_data(
#    train_inputs,
#    train_outputs,
#    gp_model,
#    likelihood,
#)



print("GP MODEL SAVED IN: ", save_path)