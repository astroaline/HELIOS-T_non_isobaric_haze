from input import *

## info for plotting ##

parameter_ranges = {}
for parameter, prior in priors.items():
    parameter_ranges[parameter] = [prior[1], prior[0]+prior[1]]

parameter_labels = {"T": r"$T$", "log_xh2o": r"$log(X_{\rm H_2O})$", "log_xhcn": r"$log(X_{\rm HCN})$",
                    "log_xnh3": r"$log(X_{\rm NH_3})$", "log_xch4": r"$log(X_{\rm CH_4})$",
                    "log_xco": r"$log(X_{\rm CO})$", "R0": r"$R_0$", "log_P0": r"$log(P_0)$",
                    "log_tau_ref": r"$log(\tau_{\rm ref})$", "Q0": r"$Q_0$", "a": r"$a$", "log_r_c": r"$log(r_c)$",
                    "log_p_cia": r"$log(P_{\rm CIA})$", "log_P_cloudtop": r"$log(P_{\rm cloudtop})$",
                    "log_cloud_depth": r"$b_c$",
                    "Rstar": r"$R_\star$", "G": r"$g$", "line": r"$flat-line$"}      # labels for all possible parameters

parameter_colors = {"T": ['Reds',0.4], "log_xh2o": ['Blues',0.4], "log_xhcn": ['Oranges',0.4],
                    "log_xnh3": ['Greens',0.4], "log_xch4": ['GnBu',0.4], "log_xco": ['YlOrBr',0.4],
                    "R0": ["PuRd",0.4], "log_P0": ["GnBu", 0.4], "log_tau_ref": ["RdPu", 0.4], "Q0": ["BuPu",0.5],
                    "a": ["YlGnBu", 0.4], "log_r_c": ["PuBu", 0.3], "log_cloud_depth": ["Blues", 0.7], "log_p_cia": ["YlOrBr", 0.4],
                    "log_P_cloudtop": ["GnBu",0.5], "Rstar": ["YlOrRd",0.4], "G": ["YlGn", 0.4], "line": ['Oranges', 0.4]}      # colours for plots
