# -*- coding: utf-8 -*-
"""
PyStan Model for Collagen Data Fitting

Created on Tue Sep  1 08:53:40 2020

@author: David Sohutskay
"""
 
import pystan
import pickle
import numpy as np
import arviz as az

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


multilevel_model_code = """
// Stan model for combined data fitting

functions {
    // Stress equations functor to pass to the algebraic solver
    vector system(vector y,vector theta,real[] x_r,int[] x_i) {   // unknowns, parameters, data (real), data (integer)
        vector[3] stress; // Define output vector
        real sigma_xx = y[1]; // We are solving for stress in x
        real lamda_yy = y[2]; // We are solving for extension in y
        real lamda_zz = y[3]; // We are solving for extension in z
        real lamda_xx = x_r[1]; // We know the extension in x, which is given as a real input
        real kv = theta[1];
        real k0 = theta[2];
        real kf = theta[3];
        real k2 = theta[4];
        real kappa = theta[5];
        real mu = theta[6];
        real phif = theta[7];
        real a0x = cos(mu);
        real a0y = sin(mu);
        real a0z = 0;

        real J = lamda_xx*lamda_yy*lamda_zz;
        real I1e = (lamda_xx*lamda_xx) + (lamda_yy*lamda_yy) + (lamda_zz*lamda_zz);
        real I4e = (a0x*a0x)*(lamda_xx*lamda_xx) + (a0y*a0y)*(lamda_yy*lamda_yy) + (a0z*a0z)*(lamda_zz*lamda_zz);
    
        real Psif = (kf/(2*k2))*exp(k2*pow((kappa*I1e + (1 - 3*kappa)*I4e - 1),2));
        real Psif1 = 2*k2*kappa*(kappa*I1e + (1 - 3*kappa)*I4e - 1)*Psif;
        real Psif4 = 2*k2*(1 - 3*kappa)*(kappa*I1e + (1 - 3*kappa)*I4e - 1)*Psif;
    
        // Define output sigma_xx, sigma_yy, sigma_zz such that the first equation has sigma_xx, the others are equal to zero
        stress[1] = (phif)*(kv*(J-1) - (2*k0/J) + (k0*lamda_xx*lamda_xx) + (2*Psif1*lamda_xx*lamda_xx) + (Psif4*a0x*a0x*lamda_xx*lamda_xx)) - sigma_xx;
        stress[2] = (phif)*(kv*(J-1) - (2*k0/J) + (k0*lamda_yy*lamda_yy) + (2*Psif1*lamda_yy*lamda_yy) + (Psif4*a0y*a0y*lamda_yy*lamda_yy));
        stress[3] = (phif)*(kv*(J-1) - (2*k0/J) + (k0*lamda_zz*lamda_zz) + (2*Psif1*lamda_zz*lamda_zz) + (Psif4*a0z*a0z*lamda_zz*lamda_zz));
        return stress;
    }
}

data {
    // Observed collagen densities
    int<lower=0> n_phif_obs;    // Number of observations for phif
    int<lower=0> n_phif_groups;     // Number of groups for phif
    real<lower=0> phif_obs[n_phif_obs];  // Observed values of phif
    int<lower=0> phif_groups[n_phif_groups];  // Number of observations in each group
    
    // Obserded fiber angle probabilities from the FFT
    int<lower=0> n_von_mises_prob_obs; // Number of observations for von_mises_prob_obs
    int<lower=0> n_von_mises_prob_groups;     // Number of groups for von_mises_prob_obs
    real<lower=0> von_mises_prob_obs[n_von_mises_prob_obs]; // Observed values of von_mises_prob_obs
    int<lower=0> von_mises_prob_groups[n_von_mises_prob_groups];  // Number of observations in each group

    // Observed angles (same number of observations and shape as above)
    real angles[n_von_mises_prob_obs]; // Observed values of fiber angle

    // Observed stress values
    int<lower=0> n_stress_obs; // Number of observations for stress
    int<lower=0> n_stress_groups;     // Number of groups for stress
    real<lower=0> stress_obs[n_stress_obs]; // Observed values of stress
    int<lower=0> stress_groups[n_stress_groups];  // Number of observations in each stress group
    
    // Observed lamda_xx (extension) values
    real<lower=0> extension_obs[n_stress_obs]; // Observed values of strain
    
    // Posterior predictive
    int<lower=0> n_stress_predicted; // number of predicted extension values
    real<lower=0> extension_predicted[n_stress_predicted]; // extension values we want predictions for
    real<lower=0> stress_mean_guess_phif_4[n_stress_predicted]; // we need a guess for the posterior solver, we can use the means
    real<lower=0> stress_mean_guess_phif_20[n_stress_predicted];
    real<lower=0> stress_mean_guess_phif_40[n_stress_predicted];
    real<lower=0> stress_mean_guess_phif_RS[n_stress_predicted];
    int<lower=0> n_von_mises_prob_predicted; // number of predicted von mises values
    real angles_predicted[n_von_mises_prob_predicted]; // angle values we want predictions for
}

transformed data { 
    // This only includes transformations that do not also use parameters and are not part of the sampling
    real x_r[1]; // We will pass in one strain value at a time
    int x_i[0]; // No ints are needed
}

parameters {
    // Density parameters
    real<lower=0> phif[n_stress_groups]; // Fit values of phif (there are four values here, even if we do not have observed)
    real<lower=0> phif_sigma[n_phif_groups]; // Uncertainty in density (just phif-4, no sigma for the unmeasured)
    
    // Microstructural parameters
    real<lower=0> b[n_von_mises_prob_groups]; // Fiber dispersion
    real<lower=-1.57079633,upper=1.57079633> mu[n_von_mises_prob_groups]; // Fiber orientation
    real<lower=0> von_mises_prob_sigma[n_von_mises_prob_groups]; // Uncertainty in fibers
    
    // Stiffness parameters
    real<lower=0> kv; // Volumetric penalty
    real<lower=0> k0; // Neo-Hookean stiffness
    real<lower=0> kf; // Fiber stiffness
    real<lower=0> k2; // Non-linear stiffness
    real<lower=0> stress_sigma[n_stress_groups]; // Uncertainty in stress measurement
}

transformed parameters {
    real<lower=0> von_mises_prob[n_von_mises_prob_obs]; // Fit values
    real<lower=0.0,upper=(1.0/3.0)> kappa[n_von_mises_prob_groups]; // Dispersion parameter
    real stress[n_stress_obs]; // Fit values <lower=0>
    
    // Special block so we can define an int index
    {
        // Define variables
        int pos_von_mises; // Index for loop
        
        // Evaluate the Von Mises probability function
        pos_von_mises = 1;
        for (group in 1:n_von_mises_prob_groups) {
            // Transform b into kappa for the stress equations (conveniently inside this loop)
            kappa[group] = (1./3.)*(1 - modified_bessel_first_kind(1,b[group])/modified_bessel_first_kind(0,b[group]));
            for (num in 1:von_mises_prob_groups[group]) {
                von_mises_prob[pos_von_mises] = exp(b[group]* cos(2*(angles[pos_von_mises] - mu[group]))) / modified_bessel_first_kind(0,b[group]);
                pos_von_mises = pos_von_mises + 1;
            }
        }
    }
    
    {
        // Define variables
        int pos_stress; // Index for loop
        vector[3] y; // Solution
        vector[3] y_guess; // Solution guess
        vector[7] theta; // Stacked parameters
        
        theta[1] = kv; // Stack the parameters that do not change
        theta[2] = k0;
        theta[3] = kf;
        theta[4] = k2;
        
        // Loop to evaluate the Gasser-Ogden-Holzapfel constitutive model
        pos_stress = 1;
        for (group in 1:n_stress_groups) {
            theta[5] = kappa[group]; // Stack the parameters that do change
            theta[6] = mu[group];
            theta[7] = phif[group];
            for (num in 1:stress_groups[group]) { 
                // Find the solution using algebra_solver_newton or algebra_solver
                // If needed, can change , rel_tol, f_tol, max_steps with additonal arguments
                //real rel_tol = 1e-10;  // default
                //real f_tol = 1e-5;  // adjusted empirically
                //int max_steps = 1000;  // default
                
                y_guess[1] = stress_obs[pos_stress]; // Changes on each iteration
                y_guess[2] = 1/(extension_obs[pos_stress]*extension_obs[pos_stress]);
                y_guess[3] = 1/(extension_obs[pos_stress]*extension_obs[pos_stress]);
                
                y = algebra_solver(system, y_guess, theta, segment(extension_obs, pos_stress, 1), x_i);

                stress[pos_stress] = y[1]; // Only need to store the stress, not the extensions/lamdas
                pos_stress = pos_stress + 1;
            }
        }
    }
    
}

model {
    // Define position vectors to be used for iteration indices
    int pos_phif;
    int pos_von_mises;
    int pos_stress;
    
    phif ~ normal(0,1);
    phif_sigma ~ exponential(1);
    b ~ normal(0,1);
    mu ~ normal(0,1);
    von_mises_prob_sigma ~ exponential(1);
    kv ~ normal(0,1);
    k0 ~ normal(0,1);
    kf ~ normal(0,1);
    k2 ~ normal(0,1);
    stress_sigma ~ exponential(1);
    
    // Fit density likelihood model
    pos_phif = 1;
    for (group in 1:n_phif_groups) {
        segment(phif_obs, pos_phif, phif_groups[group]) ~ normal(phif[group], phif_sigma[group]);
        pos_phif = pos_phif + phif_groups[group];
    }
    
    // Von Mises likelihood model
    pos_von_mises = 1;
    for (group in 1:n_von_mises_prob_groups) {
        for (num in 1:von_mises_prob_groups[group]) {
            von_mises_prob_obs[pos_von_mises] ~ normal(von_mises_prob[pos_von_mises], von_mises_prob_sigma[group]);
            pos_von_mises = pos_von_mises + 1;
        }
    }
    
    // Gasser-Ogden-Holzapfel likelihood model
    pos_stress = 1;
    for (group in 1:n_stress_groups) {
        for (num in 1:stress_groups[group]) {
            stress_obs[pos_stress] ~ normal(stress[pos_stress], stress_sigma[group]);
            pos_stress = pos_stress + 1;
        }
    }
}

generated quantities{
    // (To generate posterior predictive curves for the Von Mises distribution and Gasser-Ogden-Holzapfel stress)
    
    // Define the variables
    real stress_predicted_phif_4[n_stress_predicted];
    real stress_predicted_phif_20[n_stress_predicted];
    real stress_predicted_phif_40[n_stress_predicted];
    real stress_predicted_phif_RS[n_stress_predicted];
    real von_mises_prob_predicted_phif_4[n_von_mises_prob_predicted];
    real von_mises_prob_predicted_phif_20[n_von_mises_prob_predicted];
    real von_mises_prob_predicted_phif_40[n_von_mises_prob_predicted];
    real von_mises_prob_predicted_phif_RS[n_von_mises_prob_predicted];
    real stress_mean_predicted_phif_4[n_stress_predicted];
    real stress_mean_predicted_phif_20[n_stress_predicted];
    real stress_mean_predicted_phif_40[n_stress_predicted];
    real stress_mean_predicted_phif_RS[n_stress_predicted];
    real von_mises_prob_mean_predicted_phif_4[n_von_mises_prob_predicted];
    real von_mises_prob_mean_predicted_phif_20[n_von_mises_prob_predicted];
    real von_mises_prob_mean_predicted_phif_40[n_von_mises_prob_predicted];
    real von_mises_prob_mean_predicted_phif_RS[n_von_mises_prob_predicted];
    
    vector[7] theta_phif_4 = [kv, k0, kf, k2, kappa[1], mu[1], phif[1]]'; // Stacked parameters
    vector[7] theta_phif_20 = [kv, k0, kf, k2, kappa[2], mu[2], phif[2]]';
    vector[7] theta_phif_40 = [kv, k0, kf, k2, kappa[3], mu[3], phif[3]]';
    vector[7] theta_phif_RS = [kv, k0, kf, k2, kappa[4], mu[4], phif[4]]';
    
    // Generate scaled values (no inference)
    real<lower=0> phif_scaled[n_stress_groups];
    
    for (n in 1:n_stress_groups){
        phif_scaled[n] = phif[n]/phif[n_stress_groups]; // Divide by phifRS to get the scaled phif
    }
    
    // Evaluate the Von Mises probability function
    for (n in 1:n_von_mises_prob_predicted) {
        von_mises_prob_mean_predicted_phif_4[n] = exp(b[1]* cos(2*(angles_predicted[n] - mu[1]))) / modified_bessel_first_kind(0,b[1]);
        von_mises_prob_mean_predicted_phif_20[n] = exp(b[2]* cos(2*(angles_predicted[n] - mu[2]))) / modified_bessel_first_kind(0,b[2]);
        von_mises_prob_mean_predicted_phif_40[n] = exp(b[3]* cos(2*(angles_predicted[n] - mu[3]))) / modified_bessel_first_kind(0,b[3]);
        von_mises_prob_mean_predicted_phif_RS[n] = exp(b[4]* cos(2*(angles_predicted[n] - mu[4]))) / modified_bessel_first_kind(0,b[4]);
        
        von_mises_prob_predicted_phif_4[n] = normal_rng(von_mises_prob_mean_predicted_phif_4[n], von_mises_prob_sigma[1]);
        von_mises_prob_predicted_phif_20[n] = normal_rng(von_mises_prob_mean_predicted_phif_20[n], von_mises_prob_sigma[2]);
        von_mises_prob_predicted_phif_40[n] = normal_rng(von_mises_prob_mean_predicted_phif_40[n], von_mises_prob_sigma[3]);
        von_mises_prob_predicted_phif_RS[n] = normal_rng(von_mises_prob_mean_predicted_phif_RS[n], von_mises_prob_sigma[4]);
    }

    // Loop to evaluate the Gasser-Ogden-Holzapfel constitutive model
    for (n in 1:n_stress_predicted) {
        // Find the solution using algebra_solver_newton or algebra_solver
        // If needed, can change the solver parameters as described in the transformed variables section!
        
        // Guess changes on each iteration
        vector[3] y_guess_phif_4 = [stress_mean_guess_phif_4[n], 1/(extension_predicted[n]*extension_predicted[n]), 1/(extension_predicted[n]*extension_predicted[n])]';
        vector[3] y_guess_phif_20 = [stress_mean_guess_phif_20[n], 1/(extension_predicted[n]*extension_predicted[n]), 1/(extension_predicted[n]*extension_predicted[n])]';
        vector[3] y_guess_phif_40 = [stress_mean_guess_phif_40[n], 1/(extension_predicted[n]*extension_predicted[n]), 1/(extension_predicted[n]*extension_predicted[n])]';
        vector[3] y_guess_phif_RS = [stress_mean_guess_phif_RS[n], 1/(extension_predicted[n]*extension_predicted[n]), 1/(extension_predicted[n]*extension_predicted[n])]';
        
        // Solution also changes, we also do not need to store (but could for example also save the lamdas!)
        vector[3] y_phif_4 = algebra_solver(system, y_guess_phif_4, theta_phif_4, segment(extension_predicted, n, 1), x_i);
        vector[3] y_phif_20 = algebra_solver(system, y_guess_phif_20, theta_phif_20, segment(extension_predicted, n, 1), x_i);
        vector[3] y_phif_40 = algebra_solver(system, y_guess_phif_40, theta_phif_40, segment(extension_predicted, n, 1), x_i);
        vector[3] y_phif_RS = algebra_solver(system, y_guess_phif_RS, theta_phif_RS, segment(extension_predicted, n, 1), x_i);

        stress_mean_predicted_phif_4[n] = y_phif_4[1]; // Only need to store the stress, not the extensions/lamdas (but we could!)
        stress_mean_predicted_phif_20[n] = y_phif_20[1]; // Only need to store the stress, not the extensions/lamdas
        stress_mean_predicted_phif_40[n] = y_phif_40[1]; // Only need to store the stress, not the extensions/lamdas
        stress_mean_predicted_phif_RS[n] = y_phif_RS[1]; // Only need to store the stress, not the extensions/lamdas
        
        stress_predicted_phif_4[n] = normal_rng(stress_mean_predicted_phif_4[n], stress_sigma[1]);
        stress_predicted_phif_20[n] = normal_rng(stress_mean_predicted_phif_20[n], stress_sigma[2]);
        stress_predicted_phif_40[n] = normal_rng(stress_mean_predicted_phif_40[n], stress_sigma[3]);
        stress_predicted_phif_RS[n] = normal_rng(stress_mean_predicted_phif_RS[n], stress_sigma[4]);
    }
}
"""

# Data needs to pass in 
all_data = {'n_phif_obs': len(density_data_4) + len(density_data_20) + len(density_data_40),
            'n_phif_groups': 3,
            'phif_obs': np.concatenate((density_data_4,density_data_20,density_data_40), axis=None).tolist(),
            'phif_groups': [len(density_data_4),len(density_data_20),len(density_data_40)],
            'n_von_mises_prob_obs': len(von_mises_prob_data_4) + len(von_mises_prob_data_20) + len(von_mises_prob_data_40) + len(von_mises_prob_data_RS),
            'n_von_mises_prob_groups': 4,
            'von_mises_prob_obs': np.concatenate((von_mises_prob_data_4, von_mises_prob_data_20, von_mises_prob_data_40, von_mises_prob_data_RS), axis=None).tolist(),
            'von_mises_prob_groups': [len(von_mises_prob_data_4), len(von_mises_prob_data_20), len(von_mises_prob_data_40), len(von_mises_prob_data_RS)],
            'angles': np.concatenate((angle_data_4, angle_data_20, angle_data_40, angle_data_RS), axis=None).tolist(),
            'n_stress_obs': len(stress_data_4) + len(stress_data_20) + len(stress_data_40) + len(stress_data_RS),
            'n_stress_groups': 4,
            'stress_obs': np.concatenate((stress_data_4, stress_data_20, stress_data_40, stress_data_RS), axis=None).tolist(),
            'stress_groups': [len(stress_data_4), len(stress_data_20), len(stress_data_40), len(stress_data_RS)],
            'extension_obs': np.concatenate((extension_data_4, extension_data_20, extension_data_40, extension_data_RS), axis=None).tolist(),
            'n_stress_predicted': len(np.linspace(1,1.35,36)),
            'extension_predicted': np.linspace(1,1.35,36),
            'stress_mean_guess_phif_4': mean_stress_data_4,
            'stress_mean_guess_phif_20': mean_stress_data_20,
            'stress_mean_guess_phif_40': mean_stress_data_40,
            'stress_mean_guess_phif_RS': mean_stress_data_RS,
            'n_von_mises_prob_predicted': len(np.linspace(-np.pi/2, np.pi/2, 180)),
            'angles_predicted': np.linspace(-np.pi/2, np.pi/2, 180),
            }

multilevel_model = pystan.StanModel(model_code=multilevel_model_code)
fit = multilevel_model.sampling(data=all_data, verbose=True, iter=2000, chains=4, n_jobs=-1, sample_file = 'stanwound_sample_file.csv', init='random', init_r=0.1)

print(fit.stansummary(pars=('phif', 'phif_sigma', 'b', 'mu', 'von_mises_prob_sigma', 'kv', 'k0', 'kf', 'k2', 'stress_sigma')))

data = az.from_pystan(
    posterior=fit,
)

az.to_netcdf(data, 'save_arviz_data_stanwound')

with open('stanwound_model_pickle.pkl', 'wb') as f:
    pickle.dump(multilevel_model, f, protocol=pickle.HIGHEST_PROTOCOL)

# pandas
dataframe = fit.to_dataframe(pars=('kv', 'k0', 'kf', 'k2', 'b', 'mu', 'phif', 'phif_scaled'), permuted=True)
dataframe.to_csv('stanwound_fit_permuted.csv')

predictive_dataframe = fit.to_dataframe(pars=('stress_mean_predicted_phif_4', 'stress_predicted_phif_4',
                                   'stress_mean_predicted_phif_20', 'stress_predicted_phif_20',
                                   'stress_mean_predicted_phif_40', 'stress_predicted_phif_40',
                                   'stress_mean_predicted_phif_RS', 'stress_predicted_phif_RS',
                                   'von_mises_prob_mean_predicted_phif_4', 'von_mises_prob_predicted_phif_4',
                                   'von_mises_prob_mean_predicted_phif_20', 'von_mises_prob_predicted_phif_20',
                                   'von_mises_prob_mean_predicted_phif_40', 'von_mises_prob_predicted_phif_40',
                                   'von_mises_prob_mean_predicted_phif_RS', 'von_mises_prob_predicted_phif_RS',), permuted=True)
predictive_dataframe.to_csv('stanwound_fit_predictive_permuted.csv')

# # load it at some future point
# with open('multilevel_model.pkl', 'rb') as f:
#     multilevel_model = pickle.load(f)
