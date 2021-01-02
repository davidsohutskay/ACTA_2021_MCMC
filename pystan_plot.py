# -*- coding: utf-8 -*-
"""
PyStan Plot

Created on Wed Sep  2 09:59:32 2020

@author: David
"""

import numpy as np
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
seamap = mpl.colors.ListedColormap(sns.color_palette("husl"))

#------------------------#
# DENSITY DATA
#------------------------#

density_data_4 = np.loadtxt('density_data_4.csv', delimiter=',')
density_data_20 = np.loadtxt('density_data_20.csv', delimiter=',')
density_data_40 = np.loadtxt('density_data_40.csv', delimiter=',')

#------------------------#
# FIBER DATA
#------------------------#

von_mises_prob_data_4 = np.loadtxt('von_mises_prob_data_4.csv', delimiter=',')
angle_data_4 = np.loadtxt('angle_data_4.csv', delimiter=',')
von_mises_prob_data_20 = np.loadtxt('von_mises_prob_data_20.csv', delimiter=',')
angle_data_20 = np.loadtxt('angle_data_20.csv', delimiter=',')
von_mises_prob_data_40 = np.loadtxt('von_mises_prob_data_40.csv', delimiter=',')
angle_data_40 = np.loadtxt('angle_data_40.csv', delimiter=',')
von_mises_prob_data_RS = np.loadtxt('von_mises_prob_data_RS.csv', delimiter=',')
angle_data_RS = np.loadtxt('angle_data_RS.csv', delimiter=',')

mean_von_mises_prob_data_4 = np.loadtxt('mean_von_mises_prob_data_4.csv', delimiter=',')
mean_von_mises_prob_data_20 = np.loadtxt('mean_von_mises_prob_data_20.csv', delimiter=',')
mean_von_mises_prob_data_40 = np.loadtxt('mean_von_mises_prob_data_40.csv', delimiter=',')
mean_von_mises_prob_data_RS = np.loadtxt('mean_von_mises_prob_data_RS.csv', delimiter=',')
std_von_mises_prob_data_4 = np.loadtxt('std_von_mises_prob_data_4.csv', delimiter=',')
std_von_mises_prob_data_20 = np.loadtxt('std_von_mises_prob_data_20.csv', delimiter=',')
std_von_mises_prob_data_40 = np.loadtxt('std_von_mises_prob_data_40.csv', delimiter=',')
std_von_mises_prob_data_RS = np.loadtxt('std_von_mises_prob_data_RS.csv', delimiter=',')

#------------------------#
# MECHANICAL DATA
#------------------------#

stress_data_4 = np.loadtxt('stress_data_4.csv', delimiter=',')
extension_data_4 = np.loadtxt('extension_data_4.csv', delimiter=',')
stress_data_20 = np.loadtxt('stress_data_20.csv', delimiter=',')
extension_data_20 = np.loadtxt('extension_data_20.csv', delimiter=',')
stress_data_40 = np.loadtxt('stress_data_40.csv', delimiter=',')
extension_data_40 = np.loadtxt('extension_data_40.csv', delimiter=',')
stress_data_RS = np.loadtxt('stress_data_RS.csv', delimiter=',')
extension_data_RS = np.loadtxt('extension_data_RS.csv', delimiter=',')

mean_stress_data_4 = np.loadtxt('mean_stress_data_4.csv', delimiter=',')
mean_stress_data_20 = np.loadtxt('mean_stress_data_20.csv', delimiter=',')
mean_stress_data_40 = np.loadtxt('mean_stress_data_40.csv', delimiter=',')
mean_stress_data_RS = np.loadtxt('mean_stress_data_RS.csv', delimiter=',')
std_stress_data_4 = np.loadtxt('std_stress_data_4.csv', delimiter=',')
std_stress_data_20 = np.loadtxt('std_stress_data_20.csv', delimiter=',')
std_stress_data_40 = np.loadtxt('std_stress_data_40.csv', delimiter=',')
std_stress_data_RS = np.loadtxt('std_stress_data_RS.csv', delimiter=',')

data = az.from_netcdf('save_arviz_data_stanwound')

az.style.use("default")
    
az.rhat(data, var_names=['kv', 'k0', 'kf', 'k2','b','mu','phif'])

extra_kwargs = {"color": "lightsteelblue"}

az.plot_ess(data, kind="local", var_names=['kv', 'k0', 'kf', 'k2','b','mu','phif'], 
            figsize=(18,18), color="royalblue", extra_kwargs=extra_kwargs, textsize=20)

az.plot_ess(data, kind="quantile", var_names=['kv', 'k0', 'kf', 'k2','b','mu','phif'], 
            figsize=(18,18), color="royalblue", extra_kwargs=extra_kwargs, textsize=20)

az.plot_ess(data, kind="evolution", var_names=['kv', 'k0', 'kf', 'k2','b','mu','phif'], 
            figsize=(18,18), color="royalblue", extra_kwargs=extra_kwargs, textsize=20)

az.plot_trace(data, var_names=['kv', 'k0', 'kf', 'k2'], compact=False, combined=False,figsize=(10,6.5)) # , kind="trace"
az.plot_trace(data, var_names=['b','mu','phif'], compact=True, combined=True,figsize=(10,5)) # , kind="rank_plot" 

axes = az.plot_forest(
data,
kind="forestplot",
var_names=['kv', 'k0', 'kf','k2'],
linewidth=4,
combined=True,
ridgeplot_overlap=1.5,
colors="blue",
figsize=(4, 4),
)

axes = az.plot_forest(
data,
kind="forestplot",
var_names=['phif_scaled'],
linewidth=4,
combined=True,
ridgeplot_overlap=1.5,
colors="blue",
figsize=(4, 4),
)

axes = az.plot_forest(
data,
kind="forestplot",
var_names=['mu'],
linewidth=4,
combined=True,
ridgeplot_overlap=1.5,
colors="blue",
figsize=(4, 4),
)

axes = az.plot_forest(
data,
kind="forestplot",
var_names=['b'],
linewidth=4,
combined=True,
ridgeplot_overlap=1.5,
colors="blue",
figsize=(4, 4),
)

kde_kwargs = {'kind':'kde', 'plot_kwargs':{"linewidth": 3}, 'fill_kwargs':{'alpha':0.1}} #, "color": "black"
az.plot_pair(data,
              var_names=['kv','k0', 'kf', 'k2','phif'],
              marginals=True,
              marginal_kwargs=kde_kwargs,
              textsize = 20,
              figsize=(24,18),
              point_estimate="median",
              kind=['scatter','kde']
              )

az.plot_pair(data,
              var_names=['b','mu'],
              marginals=True,
              marginal_kwargs=kde_kwargs,
              textsize = 20,
              figsize=(24,18),
              point_estimate="median",
              kind=['scatter','kde']
              )

az.plot_autocorr(data,var_names=('kv', 'k0', 'kf', 'k2','b','mu','phif'))


# Load in permuted traces from PyStan run
predictive_df = pd.read_csv('stanwound_fit_predictive_permuted.csv')
stress_mean_predicted_phif_4 = np.empty([0,3])
stress_mean_predicted_phif_20 = np.empty([0,3])
stress_mean_predicted_phif_40 = np.empty([0,3])
stress_mean_predicted_phif_RS = np.empty([0,3])
stress_predicted_phif_4 = np.empty([0,3])
stress_predicted_phif_20 = np.empty([0,3])
stress_predicted_phif_40 = np.empty([0,3])
stress_predicted_phif_RS = np.empty([0,3])
von_mises_prob_mean_predicted_phif_4 = np.empty([0,3])
von_mises_prob_mean_predicted_phif_20 = np.empty([0,3])
von_mises_prob_mean_predicted_phif_40 = np.empty([0,3])
von_mises_prob_mean_predicted_phif_RS = np.empty([0,3])
von_mises_prob_predicted_phif_4 = np.empty([0,3])
von_mises_prob_predicted_phif_20 = np.empty([0,3])
von_mises_prob_predicted_phif_40 = np.empty([0,3])
von_mises_prob_predicted_phif_RS = np.empty([0,3])

for i in range(1,37,1):
    # Extract columns corresponding to variables
    stress_mean_predicted_phif_4 = np.append(stress_mean_predicted_phif_4, [np.percentile(predictive_df[[f"stress_mean_predicted_phif_4[{i}]"]].values,[15.9,50,84.1])], axis=0) # 15.9,50,84.1
    stress_mean_predicted_phif_20 = np.append(stress_mean_predicted_phif_20, [np.percentile(predictive_df[[f"stress_mean_predicted_phif_20[{i}]"]].values,[15.9,50,84.1])], axis=0)
    stress_mean_predicted_phif_40 = np.append(stress_mean_predicted_phif_40, [np.percentile(predictive_df[[f"stress_mean_predicted_phif_40[{i}]"]].values,[15.9,50,84.1])], axis=0)
    stress_mean_predicted_phif_RS = np.append(stress_mean_predicted_phif_RS, [np.percentile(predictive_df[[f"stress_mean_predicted_phif_RS[{i}]"]].values,[15.9,50,84.1])], axis=0)
    
    stress_predicted_phif_4 = np.append(stress_predicted_phif_4, [np.percentile(predictive_df[[f"stress_predicted_phif_4[{i}]"]].values,[15.9,50,84.1])], axis=0)
    stress_predicted_phif_20 = np.append(stress_predicted_phif_20, [np.percentile(predictive_df[[f"stress_predicted_phif_20[{i}]"]].values,[15.9,50,84.1])], axis=0)
    stress_predicted_phif_40 = np.append(stress_predicted_phif_40, [np.percentile(predictive_df[[f"stress_predicted_phif_40[{i}]"]].values,[15.9,50,84.1])], axis=0)
    stress_predicted_phif_RS = np.append(stress_predicted_phif_RS, [np.percentile(predictive_df[[f"stress_predicted_phif_RS[{i}]"]].values,[15.9,50,84.1])], axis=0)

for i in range(1,181,1):
    von_mises_prob_mean_predicted_phif_4 = np.append(von_mises_prob_mean_predicted_phif_4, [np.percentile(predictive_df[[f"von_mises_prob_mean_predicted_phif_4[{i}]"]].values,[15.9,50,84.1])], axis=0)
    von_mises_prob_mean_predicted_phif_20 = np.append(von_mises_prob_mean_predicted_phif_20, [np.percentile(predictive_df[[f"von_mises_prob_mean_predicted_phif_20[{i}]"]].values,[15.9,50,84.1])], axis=0)
    von_mises_prob_mean_predicted_phif_40 = np.append(von_mises_prob_mean_predicted_phif_40, [np.percentile(predictive_df[[f"von_mises_prob_mean_predicted_phif_40[{i}]"]].values,[15.9,50,84.1])], axis=0)
    von_mises_prob_mean_predicted_phif_RS = np.append(von_mises_prob_mean_predicted_phif_RS, [np.percentile(predictive_df[[f"von_mises_prob_mean_predicted_phif_RS[{i}]"]].values,[15.9,50,84.1])], axis=0)
    
    von_mises_prob_predicted_phif_4 = np.append(von_mises_prob_predicted_phif_4, [np.percentile(predictive_df[[f"von_mises_prob_predicted_phif_4[{i}]"]].values,[15.9,50,84.1])], axis=0)
    von_mises_prob_predicted_phif_20 = np.append(von_mises_prob_predicted_phif_20, [np.percentile(predictive_df[[f"von_mises_prob_predicted_phif_20[{i}]"]].values,[15.9,50,84.1])], axis=0)
    von_mises_prob_predicted_phif_40 = np.append(von_mises_prob_predicted_phif_40, [np.percentile(predictive_df[[f"von_mises_prob_predicted_phif_40[{i}]"]].values,[15.9,50,84.1])], axis=0)
    von_mises_prob_predicted_phif_RS = np.append(von_mises_prob_predicted_phif_RS, [np.percentile(predictive_df[[f"von_mises_prob_predicted_phif_RS[{i}]"]].values,[15.9,50,84.1])], axis=0)


# Load in permuted traces from PyStan run
trace_df = pd.read_csv('stanwound_fit_permuted.csv')

# Extract columns corresponding to variables
trace_phif_1 = trace_df[['phif[1]']].values
trace_phif_2 = trace_df[['phif[2]']].values
trace_phif_3 = trace_df[['phif[3]']].values
trace_phif_4 = trace_df[['phif[4]']].values

prior_phif_4 = np.random.normal(0.004, 0.0002, 1000)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

sns.set()
sns.set_style("white")
sns.set_palette("husl", color_codes=True)
sns.set_context("poster")

# Plot results
axes[0,0].set_title('Oligomer-4', y=1.05, fontweight="bold", fontsize=24)
axes[0,1].set_title('Oligomer-20', y=1.05, fontweight="bold", fontsize=24)
axes[0,2].set_title('Oligomer-40', y=1.05, fontweight="bold", fontsize=24)
axes[0,3].set_title('Rat Skin', y=1.05, fontweight="bold", fontsize=24)
x_data = np.linspace(-np.pi/2, np.pi/2, 180)
axes[0,0].plot(x_data,von_mises_prob_mean_predicted_phif_4[:,1], 'm--', label='Oligomer-4 Model')
axes[0,0].fill_between(x_data,von_mises_prob_mean_predicted_phif_4[:,0],von_mises_prob_mean_predicted_phif_4[:,2], color='m', alpha=0.5)
#axes[0,0].fill_between(np.linspace(1,1.35,36),von_mises_prob_predicted_phif_4[:,0],von_mises_prob_predicted_phif_4[:,2])
axes[0,1].plot(x_data,von_mises_prob_mean_predicted_phif_20[:,1], 'r--', label='Oligomer-20 Model')
axes[0,1].fill_between(x_data,von_mises_prob_mean_predicted_phif_20[:,0],von_mises_prob_mean_predicted_phif_20[:,2], color='r', alpha=0.5)
#axes[0,1].fill_between(np.linspace(1,1.35,36),von_mises_prob_predicted_phif_20[:,0],von_mises_prob_predicted_phif_20[:,2])
axes[0,2].plot(x_data,von_mises_prob_mean_predicted_phif_40[:,1], 'b--', label='Oligomer-40 Model')
axes[0,2].fill_between(x_data,von_mises_prob_mean_predicted_phif_40[:,0],von_mises_prob_mean_predicted_phif_40[:,2], color='b', alpha=0.5)
#axes[0,2].fill_between(np.linspace(1,1.35,36),von_mises_prob_predicted_phif_40[:,0],von_mises_prob_predicted_phif_40[:,2])
axes[0,3].plot(x_data,von_mises_prob_mean_predicted_phif_RS[:,1], 'g--', label='Rat Skin Model')
axes[0,3].fill_between(x_data,von_mises_prob_mean_predicted_phif_RS[:,0],von_mises_prob_mean_predicted_phif_RS[:,2], color='g', alpha=0.5)
#axes[0,3].fill_between(np.linspace(1,1.35,36),von_mises_prob_predicted_phif_RS[:,0],von_mises_prob_predicted_phif_RS[:,2])

# Plot data
axes[0,0].fill_between(x_data, mean_von_mises_prob_data_4 - std_von_mises_prob_data_4, 
                mean_von_mises_prob_data_4 + std_von_mises_prob_data_4, facecolor='m', alpha=0.5)
axes[0,0].plot(x_data, mean_von_mises_prob_data_4, 'm', label='Oligomer-4')
axes[0,1].fill_between(x_data, mean_von_mises_prob_data_20 - std_von_mises_prob_data_20, 
                mean_von_mises_prob_data_20 + std_von_mises_prob_data_20, facecolor='r', alpha=0.5)
axes[0,1].plot(x_data, mean_von_mises_prob_data_20, 'r', label='Oligomer-20')
axes[0,2].fill_between(x_data, mean_von_mises_prob_data_40 - std_von_mises_prob_data_40, 
                mean_von_mises_prob_data_40 + std_von_mises_prob_data_40, facecolor='b', alpha=0.5)
axes[0,2].plot(x_data, mean_von_mises_prob_data_40, 'b', label='Oligomer-40')
axes[0,3].fill_between(x_data, mean_von_mises_prob_data_RS - std_von_mises_prob_data_RS, 
                mean_von_mises_prob_data_RS + std_von_mises_prob_data_RS, facecolor='g', alpha=0.5)
axes[0,3].plot(x_data, mean_von_mises_prob_data_RS, 'g', label='Rat Skin')

# Plot results
x_data = np.linspace(1,1.35,36)
axes[1,0].plot(x_data,stress_mean_predicted_phif_4[:,1], 'm--', label='Oligomer-4 Model')
axes[1,0].fill_between(x_data,stress_mean_predicted_phif_4[:,0],stress_mean_predicted_phif_4[:,2], facecolor='m', alpha=0.5)
#axes[1,0].fill_between(np.linspace(1,1.35,36),stress_predicted_phif_4[:,0],stress_predicted_phif_4[:,2])
axes[1,1].plot(x_data,stress_mean_predicted_phif_20[:,1], 'r--', label='Oligomer-20 Model')
axes[1,1].fill_between(x_data,stress_mean_predicted_phif_20[:,0],stress_mean_predicted_phif_20[:,2], facecolor='r', alpha=0.5)
#axes[1,1].fill_between(np.linspace(1,1.35,36),stress_predicted_phif_20[:,0],stress_predicted_phif_20[:,2])
axes[1,2].plot(x_data,stress_mean_predicted_phif_40[:,1], 'b--', label='Oligomer-40 Model')
axes[1,2].fill_between(x_data,stress_mean_predicted_phif_40[:,0],stress_mean_predicted_phif_40[:,2], facecolor='b', alpha=0.5)
#axes[1,2].fill_between(np.linspace(1,1.35,36),stress_predicted_phif_40[:,0],stress_predicted_phif_40[:,2])
axes[1,3].plot(x_data,stress_mean_predicted_phif_RS[:,1], 'g--', label='Rat Skin Model')
axes[1,3].fill_between(x_data,stress_mean_predicted_phif_RS[:,0],stress_mean_predicted_phif_RS[:,2], facecolor='g', alpha=0.5)
#axes[1,3].fill_between(np.linspace(1,1.35,36),stress_predicted_phif_RS[:,0],stress_predicted_phif_RS[:,2])

# Plot data
axes[1,0].fill_between(x_data, mean_stress_data_4 - std_stress_data_4, 
                mean_stress_data_4 + std_stress_data_4, facecolor='m', alpha=0.5)
axes[1,0].plot(x_data, mean_stress_data_4, 'm', label='Oligomer-4')
axes[1,1].fill_between(x_data, mean_stress_data_20 - std_stress_data_20, 
                mean_stress_data_20 + std_stress_data_20, facecolor='r', alpha=0.5)
axes[1,1].plot(x_data, mean_stress_data_20, 'r', label='Oligomer-20')
axes[1,2].fill_between(x_data, mean_stress_data_40 - std_stress_data_40, 
                mean_stress_data_40 + std_stress_data_40, facecolor='b', alpha=0.5)
axes[1,2].plot(x_data, mean_stress_data_40, 'b', label='Oligomer-40')
axes[1,3].fill_between(x_data, mean_stress_data_RS - std_stress_data_RS, 
                mean_stress_data_RS + std_stress_data_RS, facecolor='g', alpha=0.5)
axes[1,3].plot(x_data, mean_stress_data_RS, 'g', label='Rat Skin')

# az.plot_dist(trace_phif_1, color='m', fill_kwargs={'alpha': 0.3}, label="Oligomer-4 Posterior", ax=axes[2,0])
# az.plot_dist(prior_phif_4, color='m', fill_kwargs={'alpha': 0.3}, label="Oligomer-4 Empirical", ax=axes[2,0])
# az.plot_dist(trace_phif_2, color='r', fill_kwargs={'alpha': 0.3}, label="Oligomer-20 Posterior", ax=axes[2,1])
# az.plot_dist(5*prior_phif_4, color='r', fill_kwargs={'alpha': 0.3}, label="Oligomer-20 Empirical", ax=axes[2,1])
# az.plot_dist(trace_phif_3, color='b', fill_kwargs={'alpha': 0.3}, label="Oligomer-40 Posterior", ax=axes[2,2])
# az.plot_dist(10*prior_phif_4, color='b', fill_kwargs={'alpha': 0.3}, label="Oligomer-40 Empirical", ax=axes[2,2])
# az.plot_dist(trace_phif_4, color='g', fill_kwargs={'alpha': 0.3}, label="Rat Skin Posterior", ax=axes[2,3])

axes[0,0].set_xlabel('Angle (Radian)', fontsize=24)
axes[0,0].set_ylabel('Relative Density', fontsize=24)
#axes[0,i].legend(loc='upper left');
axes[0,0].set_ylim([0, 3])
axes[0,0].set_xlim([-np.pi/2, np.pi/2])
axes[0,0].tick_params(axis='x',labelsize=20)
axes[0,0].tick_params(axis='y',labelsize=20)
    
axes[1,0].set_xlim([1, 1.35])
axes[1,0].set_xlabel('Extension', fontsize=24)
axes[1,0].set_ylabel('Stress (MPa)', fontsize=24)
axes[1,0].tick_params(axis='x',labelsize=20)
axes[1,0].tick_params(axis='y',labelsize=20)

for i in range(0,4):
    #axes[0,i].set_xlabel('Angle (Radian)', fontsize=24)
    #axes[0,i].set_ylabel('Relative Density', fontsize=24)
    #axes[0,i].legend(loc='upper left');
    axes[0,i].set_ylim([0, 3])
    axes[0,i].set_xlim([-np.pi/2, np.pi/2])
    axes[0,i].tick_params(axis='x',labelsize=20)
    axes[0,i].tick_params(axis='y',labelsize=20)

for i in range(0,4):
    #axes[1,i].legend(loc='upper left');
    axes[1,i].set_xlim([1, 1.35])
    #axes[1,i].set_xlabel('Extension', fontsize=24)
    #axes[1,i].set_ylabel('Stress (MPa)', fontsize=24)
    axes[1,i].tick_params(axis='x',labelsize=20)
    axes[1,i].tick_params(axis='y',labelsize=20)
    
# SMALL_SIZE = 20
# MEDIUM_SIZE = 24
# BIGGER_SIZE = 36

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.3)
plt.show()

# for i in range(0,4):
#     #axes[1,i].legend(loc='upper left');
#     #axes[2,i].set_xlim([1, 1.35])
#     axes[2,i].set_xlabel('Density (mg/cc)')
#     axes[2,i].set_ylabel('')
