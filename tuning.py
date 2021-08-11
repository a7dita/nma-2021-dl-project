#!/usr/bin/env python3

import main
import optuna

######################################################################
# TODO: implement tuning functions to search for optimal hyperparameters; for brevity make use of functions defined in main.py
######################################################################
# Quick start: search for optimal answer to arbitrary function
# based on a suggested search space. Copied from https://optuna.org/
#
# def objective(trial):
#     x = trial.suggest_uniform('x', -10, 10)
#     return (x - 2) ** 2

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

# study.best_params  # E.g. {'x': 2.002108042}
######################################################################
# PyTorch example for optimizing n_layers, hidden_size per layer
#
# For full example see https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

# import torch
# # 1. Define an objective function to be maximized.
# def objective(trial):

#     # 2. Suggest values of the hyperparameters using a trial object.
#     n_layers = trial.suggest_int('n_layers', 1, 3)
#     layers = []

#     in_features = 28 * 28
#     for i in range(n_layers):
#         out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
#         layers.append(torch.nn.Linear(in_features, out_features))
#         layers.append(torch.nn.ReLU())
#         in_features = out_features
#     layers.append(torch.nn.Linear(in_features, 10))
#     layers.append(torch.nn.LogSoftmax(dim=1))
#     model = torch.nn.Sequential(*layers).to(torch.device('cpu'))
#     ...
#     return accuracy

# # 3. Create a study object and optimize the objective function.
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
