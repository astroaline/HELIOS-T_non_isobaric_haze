import numpy as np
from input import *
import model
from priors import Priors2
import pdb

pri=Priors2()

class Priors:

    def __init__(self, ndim, nparams):

        self.ndim = ndim
        self.nparams = nparams

    def prior(self, cube, ndim, n_params):
        for param in parameters:
             
                i = parameters.index(param)

                if param == 'G':    # set gaussian priors for planet gravity
                    try:
                        cube[i]=pri.GaussianPrior(cube[i],g,g_uncertainty)
                    except:
                        cube[i]=pri.GaussianPrior(cube[i],g,(g_uperr+g_loerr)/2)
                elif param == 'Rstar':    # set gaussian priors for star radius
                    try:
                        cube[i]=pri.GaussianPrior(cube[i],rstar,rstar_uncertainty)
                    except:
                        cube[i]=pri.GaussianPrior(cube[i],rstar,(rstar_uperr+rstar_loerr)/2)
                else:
                    cube[i] = cube[i]*priors[param][0] + priors[param][1]  # set uniform priors based on values in input


    def loglike(self, cube, ndim, n_params, loglike_args):
        for param in parameters:
            i = parameters.index(param)
            parameter_dict[param] = cube[i]     # set priors

        mass_fraction = []
        for molecule in molecules:
            abundance_name = molecular_abundance_dict[molecule]
            mass_fraction.append([10**parameter_dict[abundance_name]])
        mass_fraction = np.array(mass_fraction)
        mass_total = np.sum(mass_fraction)  # Find sum of molecular abundances

        if mass_total < 1.0:
            x = model.Model(loglike_args[0], loglike_args[1], loglike_args[2], parameter_dict, loglike_args[3])
            ymodel = x.binned_model()

            loglikelihood = (-0.5 * ((ymodel - loglike_args[4]) / loglike_args[5])**2 - np.log(abs(loglike_args[5])*np.sqrt(2*np.pi))).sum() # evaluate loglikelihood

        else:
            loglikelihood = -1e30   # If sum > 1, set likelihood to very low value so nested-sampling will exclude the case


        return loglikelihood
