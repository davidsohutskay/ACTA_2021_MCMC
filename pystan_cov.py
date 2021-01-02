# -*- coding: utf-8 -*-
"""
Calculating uncertainty in contraction using
Pystan covariance matrix estimation
and local sensitivity analysis

Created on Wed Sep  2 14:09:25 2020

@author: David
"""

import numpy as np
#import arviz as az
import pandas as pd
import pickle
import pystan
import pystan.experimental
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import scipy as sp

def eval_kappa(b):
    return (1./3.)*(1 - sp.special.i1(b)/sp.special.i0(b));

# Load in permuted traces from PyStan run
trace_df = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\stanwound_fit_permuted.csv')
print('The trace dataframe is: ')
print(trace_df)
trace_df['kappa[1]'] = trace_df['b[1]'].map(eval_kappa)
trace_df['kappa[2]'] = trace_df['b[2]'].map(eval_kappa)
trace_df['kappa[3]'] = trace_df['b[3]'].map(eval_kappa)
trace_df['kappa[4]'] = trace_df['b[4]'].map(eval_kappa)

# Extract columns corresponding to variables
trace_matrix = trace_df[['kv', 'k0', 'kf', 'k2', 
                         'mu[1]', 'mu[2]', 'mu[3]', 'mu[4]',
                         'b[1]', 'b[2]', 'b[3]', 'b[4]',
                         'phif_scaled[1]', 'phif_scaled[2]', 'phif_scaled[3]', 'phif_scaled[4]']].values
trace_matrix_4 = trace_df[['kv', 'k0', 'kf', 'k2', 
                         'phif_scaled[1]', 'kappa[1]']].values
trace_matrix_20 = trace_df[['kv', 'k0', 'kf', 'k2', 
                         'phif_scaled[2]', 'kappa[2]']].values
trace_matrix_40 = trace_df[['kv', 'k0', 'kf', 'k2', 
                         'phif_scaled[3]', 'kappa[3]']].values
trace_matrix_RS = trace_df[['kv', 'k0', 'kf', 'k2', 
                         'phif_scaled[4]', 'kappa[4]']].values
print('The trace matrix is: ')
print(trace_matrix)

# Calculate mean values

percentiles_4 = np.percentile(trace_matrix_4, [2.5, 50, 97.5], axis=0)
percentiles_20 = np.percentile(trace_matrix_20, [2.5, 50, 97.5], axis=0)
percentiles_40 = np.percentile(trace_matrix_40, [2.5, 50, 97.5], axis=0)
percentiles_RS = np.percentile(trace_matrix_RS, [2.5, 50, 97.5], axis=0)
print('The HPD values for Oligomer-4 are: ')
print(percentiles_4)
print('The HPD values for Oligomer-20 are: ')
print(percentiles_20)
print('The HPD values for Oligomer-40 are: ')
print(percentiles_40)
print('The HPD values for Rat Skin are: ')
print(percentiles_RS)

# Calculate var/cov matrix
# Numpy will calculate for each row corresponding to a variable
covmatrix = np.cov(trace_matrix.transpose())
covmatrix_4 = np.cov(trace_matrix_4.transpose())
covmatrix_20 = np.cov(trace_matrix_20.transpose())
covmatrix_40 = np.cov(trace_matrix_40.transpose())
covmatrix_RS = np.cov(trace_matrix_RS.transpose())
print('The shape of the covariance matrix is: ')
print(covmatrix.shape)
print('The covariance matrix is: ')
print(covmatrix)

# Load in data
df_model = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\contraction_model_data.csv', encoding='latin1')
mean_contraction_neg = df_model['No-Fill'].values/df_model['No-Fill'].values[0]
# Load in data
df_mean_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\mean_contraction_4.csv', encoding='latin1')
df_kv_plus_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kv_plus_contraction_4.csv', encoding='latin1')
df_k0_plus_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k0_plus_contraction_4.csv', encoding='latin1')
df_kf_plus_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kf_plus_contraction_4.csv', encoding='latin1')
df_k2_plus_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k2_plus_contraction_4.csv', encoding='latin1')
df_phif_plus_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\phif_plus_contraction_4.csv', encoding='latin1')
df_kappa_plus_4 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kappa_plus_contraction_4.csv', encoding='latin1')
# Extract series
mean_contraction_4 = df_mean_4['avg(Volume)'].values/df_mean_4['avg(Volume)'].values[0]
kv_plus_contraction_4 = df_kv_plus_4['avg(Volume)'].values/df_kv_plus_4['avg(Volume)'].values[0]
k0_plus_contraction_4 = df_k0_plus_4['avg(Volume)'].values/df_k0_plus_4['avg(Volume)'].values[0]
kf_plus_contraction_4 = df_kf_plus_4['avg(Volume)'].values/df_kf_plus_4['avg(Volume)'].values[0]
k2_plus_contraction_4 = df_k2_plus_4['avg(Volume)'].values/df_k2_plus_4['avg(Volume)'].values[0]
phif_plus_contraction_4 = df_phif_plus_4['avg(Volume)'].values/df_phif_plus_4['avg(Volume)'].values[0]
kappa_plus_contraction_4 = df_kappa_plus_4['avg(Volume)'].values/df_kappa_plus_4['avg(Volume)'].values[0]
# Load in data
df_mean_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\mean_contraction_20.csv', encoding='latin1')
df_kv_plus_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kv_plus_contraction_20.csv', encoding='latin1')
df_k0_plus_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k0_plus_contraction_20.csv', encoding='latin1')
df_kf_plus_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kf_plus_contraction_20.csv', encoding='latin1')
df_k2_plus_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k2_plus_contraction_20.csv', encoding='latin1')
df_phif_plus_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\phif_plus_contraction_20.csv', encoding='latin1')
df_kappa_plus_20 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kappa_plus_contraction_20.csv', encoding='latin1')
# Extract series
mean_contraction_20 = df_mean_20['avg(Volume)'].values/df_mean_20['avg(Volume)'].values[0]
kv_plus_contraction_20 = df_kv_plus_20['avg(Volume)'].values/df_kv_plus_20['avg(Volume)'].values[0]
k0_plus_contraction_20 = df_k0_plus_20['avg(Volume)'].values/df_k0_plus_20['avg(Volume)'].values[0]
kf_plus_contraction_20 = df_kf_plus_20['avg(Volume)'].values/df_kf_plus_20['avg(Volume)'].values[0]
k2_plus_contraction_20 = df_k2_plus_20['avg(Volume)'].values/df_k2_plus_20['avg(Volume)'].values[0]
phif_plus_contraction_20 = df_phif_plus_20['avg(Volume)'].values/df_phif_plus_20['avg(Volume)'].values[0]
kappa_plus_contraction_20 = df_kappa_plus_20['avg(Volume)'].values/df_kappa_plus_20['avg(Volume)'].values[0]
# Load in data
df_mean_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\mean_contraction_40.csv', encoding='latin1')
df_kv_plus_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kv_plus_contraction_40.csv', encoding='latin1')
df_k0_plus_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k0_plus_contraction_40.csv', encoding='latin1')
df_kf_plus_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kf_plus_contraction_40.csv', encoding='latin1')
df_k2_plus_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k2_plus_contraction_40.csv', encoding='latin1')
df_phif_plus_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\phif_plus_contraction_40.csv', encoding='latin1')
df_kappa_plus_40 = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kappa_plus_contraction_40.csv', encoding='latin1')
# Extract series
mean_contraction_40 = df_mean_40['avg(Volume)'].values/df_mean_40['avg(Volume)'].values[0]
kv_plus_contraction_40 = df_kv_plus_40['avg(Volume)'].values/df_kv_plus_40['avg(Volume)'].values[0]
k0_plus_contraction_40 = df_k0_plus_40['avg(Volume)'].values/df_k0_plus_40['avg(Volume)'].values[0]
kf_plus_contraction_40 = df_kf_plus_40['avg(Volume)'].values/df_kf_plus_40['avg(Volume)'].values[0]
k2_plus_contraction_40 = df_k2_plus_40['avg(Volume)'].values/df_k2_plus_40['avg(Volume)'].values[0]
phif_plus_contraction_40 = df_phif_plus_40['avg(Volume)'].values/df_phif_plus_40['avg(Volume)'].values[0]
kappa_plus_contraction_40 = df_kappa_plus_40['avg(Volume)'].values/df_kappa_plus_40['avg(Volume)'].values[0]
# Load in data
df_mean_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\mean_contraction_40.csv', encoding='latin1')
df_kv_plus_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kv_plus_contraction_40.csv', encoding='latin1')
df_k0_plus_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k0_plus_contraction_40.csv', encoding='latin1')
df_kf_plus_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kf_plus_contraction_40.csv', encoding='latin1')
df_k2_plus_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\k2_plus_contraction_40.csv', encoding='latin1')
df_phif_plus_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\phif_plus_contraction_40.csv', encoding='latin1')
df_kappa_plus_RS = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\kappa_plus_contraction_40.csv', encoding='latin1')
# Extract series
mean_contraction_RS = df_mean_RS['avg(Volume)'].values/df_mean_RS['avg(Volume)'].values[0]
kv_plus_contraction_RS = df_kv_plus_RS['avg(Volume)'].values/df_kv_plus_RS['avg(Volume)'].values[0]
k0_plus_contraction_RS = df_k0_plus_RS['avg(Volume)'].values/df_k0_plus_RS['avg(Volume)'].values[0]
kf_plus_contraction_RS = df_kf_plus_RS['avg(Volume)'].values/df_kf_plus_RS['avg(Volume)'].values[0]
k2_plus_contraction_RS = df_k2_plus_RS['avg(Volume)'].values/df_k2_plus_RS['avg(Volume)'].values[0]
phif_plus_contraction_RS = df_phif_plus_RS['avg(Volume)'].values/df_phif_plus_RS['avg(Volume)'].values[0]
kappa_plus_contraction_RS = df_kappa_plus_RS['avg(Volume)'].values/df_kappa_plus_RS['avg(Volume)'].values[0]

# Calculate finite difference
var_4 = []
var_20 = []
var_40 = []
var_RS = []
kv_sensitivity_4 = []
k0_sensitivity_4 = []
kf_sensitivity_4 = []
k2_sensitivity_4 = []
phif_sensitivity_4 = []
kappa_sensitivity_4 = []
kv_sensitivity_20 = []
k0_sensitivity_20 = []
kf_sensitivity_20 = []
k2_sensitivity_20 = []
phif_sensitivity_20 = []
kappa_sensitivity_20 = []
kv_sensitivity_40 = []
k0_sensitivity_40 = []
kf_sensitivity_40 = []
k2_sensitivity_40 = []
phif_sensitivity_40 = []
kappa_sensitivity_40 = []
kv_sensitivity_RS = []
k0_sensitivity_RS = []
kf_sensitivity_RS = []
k2_sensitivity_RS = []
phif_sensitivity_RS = []
kappa_sensitivity_RS = []

# Loop over all time
for i in range(0,len(mean_contraction_4)):
    # Calculate finite difference
    kv_gradient = (mean_contraction_4[i] - kv_plus_contraction_4[i])/(0.05*percentiles_4[2,0])
    kv_sensitivity_4.append(np.sqrt(covmatrix_4[0,0])*kv_gradient)
    k0_gradient = (k0_plus_contraction_4[i] - mean_contraction_4[i])/(0.05*percentiles_4[2,1])
    k0_sensitivity_4.append(np.sqrt(covmatrix_4[1,1])*k0_gradient)
    kf_gradient = (kf_plus_contraction_4[i] - mean_contraction_4[i])/(0.05*percentiles_4[2,2])
    kf_sensitivity_4.append(np.sqrt(covmatrix_4[2,2])*kf_gradient)
    k2_gradient = (k2_plus_contraction_4[i] - mean_contraction_4[i])/(0.05*percentiles_4[2,3])
    k2_sensitivity_4.append(np.sqrt(covmatrix_4[3,3])*k2_gradient)
    phif_gradient = (phif_plus_contraction_4[i] - mean_contraction_4[i])/(0.05*percentiles_4[2,4])
    phif_sensitivity_4.append(np.sqrt(covmatrix_4[4,4])*phif_gradient)
    kappa_gradient = (kappa_plus_contraction_4[i] - mean_contraction_4[i])/(0.05*percentiles_4[2,5])
    kappa_sensitivity_4.append(np.sqrt(covmatrix_4[5,5])*kappa_gradient)
    # Assemble into vector
    sensitivity_vector_4 = np.array([kv_gradient, k0_gradient, kf_gradient,
                                   k2_gradient, phif_gradient, kappa_gradient])
    
    # Calculate finite difference
    kv_gradient = (mean_contraction_20[i] - kv_plus_contraction_20[i])/(0.05*percentiles_20[2,0])
    kv_sensitivity_20.append(np.sqrt(covmatrix_40[0,0])*kv_gradient)
    k0_gradient = (k0_plus_contraction_20[i] - mean_contraction_20[i])/(0.05*percentiles_20[2,1])
    k0_sensitivity_20.append(np.sqrt(covmatrix_40[1,1])*k0_gradient)
    kf_gradient = (kf_plus_contraction_20[i] - mean_contraction_20[i])/(0.05*percentiles_20[2,2])
    kf_sensitivity_20.append(np.sqrt(covmatrix_40[2,2])*kf_gradient)
    k2_gradient = (k2_plus_contraction_20[i] - mean_contraction_20[i])/(0.05*percentiles_20[2,3])
    k2_sensitivity_20.append(np.sqrt(covmatrix_40[3,3])*k2_gradient)
    phif_gradient = (phif_plus_contraction_20[i] - mean_contraction_20[i])/(0.05*percentiles_20[2,4])
    phif_sensitivity_20.append(np.sqrt(covmatrix_40[4,4])*phif_gradient)
    kappa_gradient = (kappa_plus_contraction_20[i] - mean_contraction_20[i])/(0.05*percentiles_20[2,5])
    kappa_sensitivity_20.append(np.sqrt(covmatrix_40[5,5])*kappa_gradient)
    # Assemble into vector
    sensitivity_vector_20 = np.array([kv_gradient, k0_gradient, kf_gradient,
                                   k2_gradient, phif_gradient, kappa_gradient])

    # Calculate finite difference
    kv_gradient = (mean_contraction_40[i] - kv_plus_contraction_40[i])/(0.05*percentiles_40[2,0])
    kv_sensitivity_40.append(np.sqrt(covmatrix_40[0,0])*kv_gradient)
    k0_gradient = (k0_plus_contraction_40[i] - mean_contraction_40[i])/(0.05*percentiles_40[2,1])
    k0_sensitivity_40.append(np.sqrt(covmatrix_40[1,1])*k0_gradient)
    kf_gradient = (kf_plus_contraction_40[i] - mean_contraction_40[i])/(0.05*percentiles_40[2,2])
    kf_sensitivity_40.append(np.sqrt(covmatrix_40[2,2])*kf_gradient)
    k2_gradient = (k2_plus_contraction_40[i] - mean_contraction_40[i])/(0.05*percentiles_40[2,3])
    k2_sensitivity_40.append(np.sqrt(covmatrix_40[3,3])*k2_gradient)
    phif_gradient = (phif_plus_contraction_40[i] - mean_contraction_40[i])/(0.05*percentiles_40[2,4])
    phif_sensitivity_40.append(np.sqrt(covmatrix_40[4,4])*phif_gradient)
    kappa_gradient = (kappa_plus_contraction_40[i] - mean_contraction_40[i])/(0.05*percentiles_40[2,5])
    kappa_sensitivity_40.append(np.sqrt(covmatrix_40[5,5])*kappa_gradient)
    # Assemble into vector
    sensitivity_vector_40 = np.array([kv_gradient, k0_gradient, kf_gradient,
                                   k2_gradient, phif_gradient, kappa_gradient])
    
    # Calculate finite difference
    kv_gradient = (mean_contraction_RS[i] - kv_plus_contraction_RS[i])/(0.05*percentiles_RS[2,0])
    kv_sensitivity_RS.append(np.sqrt(covmatrix_RS[0,0])*kv_gradient)
    k0_gradient = (k0_plus_contraction_RS[i] - mean_contraction_RS[i])/(0.05*percentiles_RS[2,1])
    k0_sensitivity_RS.append(np.sqrt(covmatrix_RS[1,1])*k0_gradient)
    kf_gradient = (kf_plus_contraction_RS[i] - mean_contraction_RS[i])/(0.05*percentiles_RS[2,2])
    kf_sensitivity_RS.append(np.sqrt(covmatrix_RS[2,2])*kf_gradient)
    k2_gradient = (k2_plus_contraction_RS[i] - mean_contraction_RS[i])/(0.05*percentiles_RS[2,3])
    k2_sensitivity_RS.append(np.sqrt(covmatrix_RS[3,3])*k2_gradient)
    phif_gradient = (phif_plus_contraction_RS[i] - mean_contraction_RS[i])/(0.05*percentiles_RS[2,4])
    phif_sensitivity_RS.append(np.sqrt(covmatrix_RS[4,4])*phif_gradient)
    kappa_gradient = (kappa_plus_contraction_RS[i] - mean_contraction_RS[i])/(0.05*percentiles_RS[2,5])
    kappa_sensitivity_RS.append(np.sqrt(covmatrix_RS[5,5])*kappa_gradient)
    # Assemble into vector
    sensitivity_vector_RS = np.array([kv_gradient, k0_gradient, kf_gradient,
                                   k2_gradient, phif_gradient, kappa_gradient])
    
    # Multiply by cov and store
    var_4.append(np.sqrt(sensitivity_vector_4.transpose().dot(covmatrix_4.dot(sensitivity_vector_4))))
    var_20.append(np.sqrt(sensitivity_vector_20.transpose().dot(covmatrix_20.dot(sensitivity_vector_20))))
    var_40.append(np.sqrt(sensitivity_vector_40.transpose().dot(covmatrix_40.dot(sensitivity_vector_40))))
    var_RS.append(np.sqrt(sensitivity_vector_RS.transpose().dot(covmatrix_RS.dot(sensitivity_vector_RS))))
    
df = pd.read_csv(r'C:\Users\David\Documents\woundpymc3\contraction_data.csv', encoding='latin1')

meanpl = df.groupby(['Group','Time'])['Area Change'].mean()
stdpl = df.groupby(['Group','Time'])['Area Change'].std()

negative = meanpl['0'].values
oligo4 = meanpl['4'].values
oligo20 = meanpl['20'].values
oligo40 = meanpl['40'].values
RS = meanpl['RS'].values

negativestd = stdpl['0'].values
negativestd[0] = 0
oligo4std = stdpl['4'].values
oligo4std[0] = 0
oligo20std = stdpl['20'].values
oligo20std[0] = 0
oligo40std = stdpl['40'].values
oligo40std[0] = 0
RSstd = stdpl['RS'].values
RSstd[0]= 0

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

sns.set()
sns.set_style("white")
sns.set_palette("husl", color_codes=True)
sns.set_context("poster")

axes[0].set_title('Experimental', y=1.05, fontweight="bold")
axes[1].set_title('Computational', y=1.05, fontweight="bold")

l1 = axes[0].errorbar([0,168,335], negative, yerr = negativestd, fmt='y', capsize=5, label='Negative')
axes[0].fill_between([0,168,335], negative - negativestd, negative + negativestd, color='y', alpha=0.5, label='Negative')
l2 = axes[0].errorbar([0,168,335], oligo4, yerr = oligo4std, fmt='m', capsize=5, label='Oligomer-4')
axes[0].fill_between([0,168,335], oligo4 - oligo4std, oligo4 + oligo4std, color='m', alpha=0.5, label='Oligomer-4')
l3 = axes[0].errorbar([0,168,335], oligo20, yerr = oligo20std, fmt='r', capsize=5, label='Oligomer-20')
axes[0].fill_between([0,168,335], oligo20 - oligo20std, oligo20 + oligo20std, color='r', alpha=0.5, label='Oligomer-20')
l4 = axes[0].errorbar([0,168,335], oligo40, yerr = oligo40std, fmt='b', capsize=5, label='Oligomer-40')
axes[0].fill_between([0,168,335], oligo40 - oligo40std, oligo40 + oligo40std, color='b', alpha=0.5, label='Oligomer-40')
l5 = axes[0].errorbar([0,168,335], RS, yerr = RSstd, fmt='g', capsize=5, label='Rat Skin')
axes[0].fill_between([0,168,335], RS - RSstd, RS + RSstd, color='g', alpha=0.5, label='Rat Skin')

axes[0].set_ylim(0,1.2)
axes[0].set_xlim(0,336)
axes[0].set_xlabel('Time (Hours)')
axes[0].set_ylabel('Area')
#axes[0].legend(loc='lower left')

time_model = np.linspace(0,335,336)
negative_model = df_model['No-Fill'].values/df_model['No-Fill'].values[0]
oligo4_model = df_model['Oligomer-4'].values/df_model['Oligomer-4'].values[0]
oligo20_model = df_model['Oligomer-20'].values/df_model['Oligomer-20'].values[0]
RS_model = df_model['Rat Skin'].values/df_model['Rat Skin'].values[0]

l6 = axes[1].plot(time_model, negative_model, 'y', label='Negative Model') # , linewidth=3,alpha=0.5
plt.fill_between(np.linspace(0,335,336), oligo4_model + var_4, oligo4_model - var_4, facecolor='m', alpha=0.5) #
l7 = axes[1].plot(time_model, oligo4_model, 'm', label='Oligomer-4 Model')
plt.fill_between(np.linspace(0,335,336), oligo20_model + var_20, oligo20_model - var_20, facecolor='r', alpha=0.5) #
l7 = axes[1].plot(time_model, oligo20_model, 'r', label='Oligomer-20 Model')
plt.fill_between(np.linspace(0,335,336), mean_contraction_40 + var_40, mean_contraction_40 - var_40, facecolor='b', alpha=0.5) #
l7 = axes[1].plot(time_model, mean_contraction_40, 'b', label='Oligomer-40 Model')
plt.fill_between(np.linspace(0,335,336), RS_model + var_RS, RS_model - var_RS, facecolor='g', alpha=0.5) #
l8 = axes[1].plot(time_model, RS_model, 'g', label='Rat Skin Model')

axes[1].set_ylim(0,1.2)
axes[1].set_xlim(0,336)
axes[1].set_xlabel('Time (Hours)')
axes[1].set_ylabel('Area')
#axes[1].legend(loc='lower left')
plt.subplots_adjust(wspace = 0.3)

# Labels to use in the legend for each line
line_labels = ["No-Fill", "Oligomer-4", "Oligomer-20", "Oligomer-40", "Rat Skin"]
# line_labels = ["Negative Data", "Oligomer-4 Data", "Oligomer-20 Data", "Oligomer-40 Data", "Rat Skin Data",
#                "Negative Model", "Oligomer-4 Model", "Oligomer-20 Model", "Oligomer-40 Model", "Rat Skin Model"]

# Create the legend
fig.legend([l1, l2, l3, l4, l5, l6, l7, l8],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Legend"  # Title for the legend
           )

# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
plt.subplots_adjust(right=0.8)

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(35, 5))

sns.set()
sns.set_style("white")
sns.set_palette("husl", color_codes=True)
sns.set_context("poster")

# axes[0].set_title('Experimental', y=1.05, fontweight="bold")
# axes[1].set_title('Computational', y=1.05, fontweight="bold")

axes[0].errorbar([0,168,335], negative, yerr = negativestd, fmt='y', capsize=5, label='Negative')
axes[0].fill_between([0,168,335], negative - negativestd, negative + negativestd, color='y', alpha=0.5, label='Negative')
axes[0].plot(time_model, negative_model, 'y', label='Negative Model') # , linewidth=3,alpha=0.5

axes[1].errorbar([0,168,335], oligo4, yerr = oligo4std, fmt='m', capsize=5, label='Oligomer-4')
axes[1].fill_between([0,168,335], oligo4 - oligo4std, oligo4 + oligo4std, color='m', alpha=0.5, label='Oligomer-4')
axes[1].fill_between(np.linspace(0,335,336), oligo4_model + var_4, oligo4_model - var_4, facecolor='m', alpha=0.5) #
axes[1].plot(time_model, oligo4_model, 'm', label='Oligomer-4 Model')

axes[2].errorbar([0,168,335], oligo20, yerr = oligo20std, fmt='r', capsize=5, label='Oligomer-20')
axes[2].fill_between([0,168,335], oligo20 - oligo20std, oligo20 + oligo20std, color='r', alpha=0.5, label='Oligomer-20')
axes[2].fill_between(np.linspace(0,335,336), oligo20_model + var_20, oligo20_model - var_20, facecolor='r', alpha=0.5) #
axes[2].plot(time_model, oligo20_model, 'r', label='Oligomer-20 Model')

axes[3].errorbar([0,168,335], oligo40, yerr = oligo40std, fmt='b', capsize=5, label='Oligomer-40')
axes[3].fill_between([0,168,335], oligo40 - oligo40std, oligo40 + oligo40std, color='b', alpha=0.5, label='Oligomer-40')
axes[3].fill_between(np.linspace(0,335,336), mean_contraction_40 + var_40, mean_contraction_40 - var_40, facecolor='b', alpha=0.5) #
axes[3].plot(time_model, mean_contraction_40, 'b', label='Oligomer-40 Model')

axes[4].errorbar([0,168,335], RS, yerr = RSstd, fmt='g', capsize=5, label='Rat Skin')
axes[4].fill_between([0,168,335], RS - RSstd, RS + RSstd, color='g', alpha=0.5, label='Rat Skin')
axes[4].fill_between(np.linspace(0,335,336), RS_model + var_RS, RS_model - var_RS, facecolor='g', alpha=0.5) #
axes[4].plot(time_model, RS_model, 'g', label='Rat Skin Model')

for i in range(5):
    axes[i].set_ylim(0,1.2)
    axes[i].set_xlim(0,336)
    axes[i].set_xlabel('Time (Hours)')
    axes[i].set_ylabel('Area')
    #axes[i].legend(loc='lower left')

plt.subplots_adjust(wspace = 0.3)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(35, 35))

sns.set()
sns.set_style("white")
sns.set_palette("husl", color_codes=True)
sns.set_context("poster")

# axes[0].set_title('Experimental', y=1.05, fontweight="bold")
# axes[1].set_title('Computational', y=1.05, fontweight="bold")

axes.errorbar([0,168,335], negative, yerr = negativestd, fmt='y', capsize=5, label='Negative')
axes.fill_between([0,168,335], negative - negativestd, negative + negativestd, color='y', alpha=0.5, label='Negative')
axes.plot(time_model, negative_model, 'y', label='Negative Model') # , linewidth=3,alpha=0.5

axes.errorbar([0,168,335], oligo4, yerr = oligo4std, fmt='m', capsize=5, label='Oligomer-4')
axes.fill_between([0,168,335], oligo4 - oligo4std, oligo4 + oligo4std, color='m', alpha=0.5, label='Oligomer-4')
axes.fill_between(np.linspace(0,335,336), oligo4_model + var_4, oligo4_model - var_4, facecolor='m', alpha=0.5) #
axes.plot(time_model, oligo4_model, 'm', label='Oligomer-4 Model')

axes.errorbar([0,168,335], oligo20, yerr = oligo20std, fmt='r', capsize=5, label='Oligomer-20')
axes.fill_between([0,168,335], oligo20 - oligo20std, oligo20 + oligo20std, color='r', alpha=0.5, label='Oligomer-20')
axes.fill_between(np.linspace(0,335,336), oligo20_model + var_20, oligo20_model - var_20, facecolor='r', alpha=0.5) #
axes.plot(time_model, oligo20_model, 'r', label='Oligomer-20 Model')

axes.errorbar([0,168,335], oligo40, yerr = oligo40std, fmt='b', capsize=5, label='Oligomer-40')
axes.fill_between([0,168,335], oligo40 - oligo40std, oligo40 + oligo40std, color='b', alpha=0.5, label='Oligomer-40')
axes.fill_between(np.linspace(0,335,336), mean_contraction_40 + var_40, mean_contraction_40 - var_40, facecolor='b', alpha=0.5) #
axes.plot(time_model, mean_contraction_40, 'b', label='Oligomer-40 Model')

axes.errorbar([0,168,335], RS, yerr = RSstd, fmt='g', capsize=5, label='Rat Skin')
axes.fill_between([0,168,335], RS - RSstd, RS + RSstd, color='g', alpha=0.5, label='Rat Skin')
axes.fill_between(np.linspace(0,335,336), RS_model + var_RS, RS_model - var_RS, facecolor='g', alpha=0.5) #
axes.plot(time_model, RS_model, 'g', label='Rat Skin Model')

axes.set_ylim(0,1.2)
axes.set_xlim(0,336)
axes.set_xlabel('Time (Hours)')
axes.set_ylabel('Area')
#axes.legend(loc='lower left')

plt.subplots_adjust(wspace = 0.3)
plt.show()
# Plot sensitivities

# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 30))

# sns.set()
# sns.set_style("white")
# sns.set_palette("husl", color_codes=True)
# sns.set_context("poster")

# axes[0].plot(time_model, kv_sensitivity_4, label='kv Sensitivity')
# axes[0].plot(time_model, k0_sensitivity_4, label='k0 Sensitivity')
# axes[0].plot(time_model, kf_sensitivity_4, label='kf Sensitivity')
# axes[0].plot(time_model, k2_sensitivity_4, label='k2 Sensitivity')
# axes[0].plot(time_model, kappa_sensitivity_4, label='kappa Sensitivity')
# axes[0].plot(time_model, phif_sensitivity_4, label='phif Sensitivity')

# axes[1].plot(time_model, kv_sensitivity_20, label='kv_sensitivity')
# axes[1].plot(time_model, k0_sensitivity_20, label='k0_sensitivity')
# axes[1].plot(time_model, kf_sensitivity_20, label='kf_sensitivity')
# axes[1].plot(time_model, k2_sensitivity_20, label='k2_sensitivity')
# axes[1].plot(time_model, kappa_sensitivity_20, label='kappa_sensitivity')
# axes[1].plot(time_model, phif_sensitivity_20, label='phif_sensitivity')

# axes[2].plot(time_model, kv_sensitivity_40, label='kv_sensitivity')
# axes[2].plot(time_model, k0_sensitivity_40, label='k0_sensitivity')
# axes[2].plot(time_model, kf_sensitivity_40, label='kf_sensitivity')
# axes[2].plot(time_model, k2_sensitivity_40, label='k2_sensitivity')
# axes[2].plot(time_model, kappa_sensitivity_40, label='kappa_sensitivity')
# axes[2].plot(time_model, phif_sensitivity_40, label='phif_sensitivity')

# axes[3].plot(time_model, kv_sensitivity_RS, label='kv_sensitivity')
# axes[3].plot(time_model, k0_sensitivity_RS, label='k0_sensitivity')
# axes[3].plot(time_model, kf_sensitivity_RS, label='kf_sensitivity')
# axes[3].plot(time_model, k2_sensitivity_RS, label='k2_sensitivity')
# axes[3].plot(time_model, kappa_sensitivity_RS, label='kappa_sensitivity')
# axes[3].plot(time_model, phif_sensitivity_RS, label='phif_sensitivity')

# # Create legend & Show graphic
# axes[0].legend()
# axes[0].set_xlabel('Time (Hours)')
# axes[0].set_ylabel('Oligomer-4')
# axes[1].set_xlabel('Time (Hours)')
# axes[1].set_ylabel('Oligomer-20')
# axes[2].set_xlabel('Time (Hours)')
# axes[2].set_ylabel('Oligomer-40')
# axes[3].set_xlabel('Time (Hours)')
# axes[3].set_ylabel('Rat Skin')
# plt.show()

# Plot max sensitivities

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

sns.set()
sns.set_style("white")
sns.set_palette("husl", color_codes=True)
sns.set_context("poster")

barWidth = 0.2         # the width of the bars
 
# set height of bar
bars_neg = [0, 0, 0,
            0, 0, 0]
bars_4 = [max(kv_sensitivity_4), max(k0_sensitivity_4), max(kf_sensitivity_4), 
         max(k2_sensitivity_4), max(kappa_sensitivity_4), max(phif_sensitivity_4)]
bars_20 = [max(kv_sensitivity_20), max(k0_sensitivity_20), max(kf_sensitivity_20), 
         max(k2_sensitivity_20), max(kappa_sensitivity_20), max(phif_sensitivity_20)]
bars_40 = [max(kv_sensitivity_40), max(k0_sensitivity_40), max(kf_sensitivity_40), 
         max(k2_sensitivity_40), max(kappa_sensitivity_40), max(phif_sensitivity_40)]
bars_RS = [max(kv_sensitivity_RS), max(k0_sensitivity_RS), max(kf_sensitivity_RS), 
          max(k2_sensitivity_RS), max(kappa_sensitivity_RS), max(phif_sensitivity_RS)]

# Set position of bar on X axis
r0 = np.arange(len(bars_4))
r1 = [x + barWidth for x in r0]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
 
# Make the plot
#l0 = axes.bar(r0, bars_neg, color='y', width=barWidth, edgecolor='white', label='No-Fill')
l1 = axes.bar(r0, bars_4, color='m', width=barWidth, edgecolor='white', label='Oligomer-4')
l2 = axes.bar(r1, bars_20, color='r', width=barWidth, edgecolor='white', label='Oligomer-20')
l3 = axes.bar(r2, bars_40, color='b', width=barWidth, edgecolor='white', label='Oligomer-40')
l4 = axes.bar(r3, bars_RS, color='g', width=barWidth, edgecolor='white', label='Rat Skin')
# axes.set_yscale('log')

# # Labels to use in the legend for each line
# line_labels = ["No-Fill", "Oligomer-4", "Oligomer-20", "Oligomer-40", "Rat Skin"]
# # line_labels = ["Negative Data", "Oligomer-4 Data", "Oligomer-20 Data", "Oligomer-40 Data", "Rat Skin Data",
# #                "Negative Model", "Oligomer-4 Model", "Oligomer-20 Model", "Oligomer-40 Model", "Rat Skin Model"]

# # Create the legend
# fig.legend([l0, l1, l2, l3, l4],     # The line objects
#            labels=line_labels,   # The labels for each line
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,    # Small spacing around legend box
#            title="Legend"  # Title for the legend
#            )

# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
plt.subplots_adjust(right=0.7)

# Add xticks on the middle of the group bars
#plt.xlabel('group', fontweight='bold')
plt.ylabel('Max Scaled Sensitivity') #, fontweight='bold'
plt.xticks([r + barWidth for r in range(len(bars_4))], ['kv', 'k0', 'kf', 'k2', 'κ', 'ɸ'])

# Create legend & Show graphic
#plt.legend()
plt.show()