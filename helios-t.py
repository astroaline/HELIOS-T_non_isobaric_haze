import pymultinest
import numpy as np
from input import *
from asthetics import *
import data_setup
import ns_setup
import model
from matplotlib import pyplot as plt
import cornerplot
import spectrum
import time
import json
import os
import csv
import pdb
import pandas as pd

start = time.time()

print('\n\n\n' + 'now running .................... ' + planet_name + ' ' + approach_name + ' ' + model_name + '\n\n\n')

planet_path = os.environ['HOME'] + '/Desktop/PhD/CASE STUDY RAYLEIGH HAZE/HELIOS-T-master_[' + approach_name + ']/planets/' + planet_name + '/'

n_params = len(parameters)

output_directory = 'out/'
os.makedirs(os.path.dirname(output_directory), exist_ok=True)

x, x_full, integral_dict, bin_indices, ydata, yerr, wavelength_centre, wavelength_err = data_setup.data()
len_x = len(x)


# pdb.set_trace()

# Run PyMultinest ##
b = ns_setup.Priors(1, n_params)
pymultinest.run(b.loglike, b.prior, n_params,
		loglike_args=[len_x, x_full, bin_indices, integral_dict, ydata, yerr],
                outputfiles_basename=output_directory + planet_name + '_',
                resume=False, verbose=False,
                n_live_points=live)   # verbose is False if you do not want to print information while running code


json.dump(parameters, open(output_directory+planet_name+'_params.json', 'w'))  # save parameter names

a = pymultinest.Analyzer(outputfiles_basename=output_directory+planet_name+'_', n_params=n_params)
samples = a.get_equal_weighted_posterior()[:, :-1]
bestfit_params = a.get_best_fit()
#stats = a.get_stats()


## set up results ##
retrieved_results = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))

plot_percentiles = []
new_param_dict = parameter_dict
retrieved_parameters_full = {}
retrieved_parameters_list = []
for i in range(n_params):
    retrieved_parameters_full[parameters[i]] = retrieved_results[i]
    new_param_dict[parameters[i]] = bestfit_params['parameters'][i]
    # new_param_dict[parameters[i]] = retrieved_results[i][0]
    retrieved_parameters_list.append(retrieved_results[i][0])
    plot_percentiles.append(retrieved_results[i])


print("""PyMultinest result:""")
for param in parameters:
    print(param,' = ',retrieved_parameters_full[param][0],' +',retrieved_parameters_full[param][1],' -',retrieved_parameters_full[param][2])

a_lnZ = a.get_stats()['global evidence']
log_evidence = a_lnZ/np.log(10)

print
print ('************************')
print ('MAIN RESULT: Evidence Z ')
print ('************************')
print ('  log Z for model = %.1f' % (log_evidence))
print


## save retrieved results ##
with open(planet_path + 'RETRIEVED RESULTS/retrieved_results_' + planet_name + '_' + approach_name + '_' + model_name + '.txt', 'w') as save_retrieved_results:
    for param in parameters:
        save_retrieved_results.write(str(param)+' = '+str(retrieved_parameters_full[param][0])+' + '+str(retrieved_parameters_full[param][1])+' - '+str(retrieved_parameters_full[param][2])+'\n')
    save_retrieved_results.write('\n')
    save_retrieved_results.write('log Z for model = %.1f' % (log_evidence))


if model_name == 'flat_line':
    pass

else:
    ## compute model for retrieved results ##
    y = model.Model(len_x, x_full, bin_indices, new_param_dict, integral_dict)
    yfit = y.transit_depth()
    yfit_binned = y.binned_model()

    wavenumber_min = int(1e4 / wavelength_bins[-1])
    wavenumber_max = int(1e4 / wavelength_bins[0])

    index_min = int((wavenumber_min) / res)
    if res == 2:
        index_max = int((wavenumber_max) / res) - 1
    else:
        index_max = int((wavenumber_max) / res)
    step_size = int(res/0.01)


    ## change to wavelength ##
    if wavenumber is True:
        x_full_plot = 1.0 / (x_full * 1e-4)  # convert to wavelength
    else:
        x_full_plot = x_full


    ## save modeled spectrum ##
    df = pd.DataFrame({'transit_depth': yfit, 'wavelength': x_full_plot})
    df.to_csv(planet_path + 'MODELED SPECTRA RESULTS/modeled_spectrum_results_' + planet_name + '_' + approach_name + '_' + model_name + '.txt', index=False, sep=' ', header=True)


    ## plot posteriors ##
    retrieved_parameter_labels = []
    posterior_ranges = []
    color_list = []
    for param in parameters:
        posterior_ranges.append(parameter_ranges[param])
        retrieved_parameter_labels.append(parameter_labels[param])
        color_list.append(parameter_colors[param])


    # posterior_ranges[3] = [0.7, 0.81]
    # pdb.set_trace()
    for i in range(len(parameters)):
        posterior_ranges[i][0] = np.minimum(posterior_ranges[i][0], min(samples[:,i]))
        posterior_ranges[i][1] = np.maximum(posterior_ranges[i][1], max(samples[:,i]))

    fig = cornerplot.corner(samples, x_full_plot, wavelength_centre, yfit, yfit_binned, ydata, wavelength_err, yerr, posterior_ranges, color_list,
                            labels=retrieved_parameter_labels, truths=plot_percentiles, max_n_ticks=3, label_kwargs={"fontsize": 34})
    fig2 = spectrum.spec(samples, x_full_plot, wavelength_centre, yfit, yfit_binned, ydata, wavelength_err, yerr, posterior_ranges, color_list,
                            labels=retrieved_parameter_labels, truths=plot_percentiles, max_n_ticks=3, label_kwargs={"fontsize": 34})

    fig_name = 'cornerplot_' + planet_name + '_' + approach_name + '_' + model_name
    fig.savefig(planet_path + 'CORNERPLOTS/' + fig_name + '.png', format='png')
    fig.savefig(planet_path + 'CORNERPLOTS/' + fig_name + '.pdf', format='pdf')
    os.chdir(planet_path + 'CORNERPLOTS/')
    os.system('pdftops -eps ' + fig_name + '.pdf')

    fig2_name = 'spectrum_' + planet_name + '_' + approach_name + '_' + model_name
    fig2.savefig(planet_path + 'SPECTRA/' + fig2_name + '.png', bbox_inches='tight', pad_inches=0.5, format='png')
    fig2.savefig(planet_path + 'SPECTRA/' + fig2_name + '.pdf', bbox_inches='tight', pad_inches=0.5, format='pdf')
    os.chdir(planet_path + 'SPECTRA/')
    os.system('pdftops -eps ' + fig2_name + '.pdf')


end = time.time()

print("time in secs = ",end-start)
print("time in mins = ",(end-start)/60)
print("time in hours = ",(end-start)/3600)
