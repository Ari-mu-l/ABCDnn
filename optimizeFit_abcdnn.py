# python3 optimizeFit_abcdnn.py -f skewNorm_cubic -c case14 --doValidation
import os
import numpy as np
from ROOT import *
from argparse import ArgumentParser

gStyle.SetOptFit(1)
gROOT.SetBatch(True) # suppress histogram display

# define the fit function
# f(x) = 2 pdf(x) * cdf(a x)
# f(x) = 2/w pdf((x-xi)/2) * cdf(a*(x-xi)/2)
# f(x) = f(x) + ax^2 + bx + c

parser = ArgumentParser()
parser.add_argument("-f", "--fitType" , default="skewNorm_cubic")
parser.add_argument("-c", "--case"    , default="case14")
parser.add_argument("--doValidation", action = "store_true")

fitType = parser.parse_args().fitType
case = parser.parse_args().case

binlo = 400
binhi = 2500
bins = 42
print(f'Fitting hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root...')

plotDir = f'fit_plots/{case}_{fitType}_{binlo}to{binhi}_{bins}'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

if fitType=="skewNorm_quadratic":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x"
    nparams = 7
    fit     = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.00001, 0.00001, 0.0000001)
elif fitType=="skewNorm_cubic":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x"
    nparams = 8
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)
elif fitType=="skewNorm_4":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x + [8] * x * x * x * x"
    nparams = 9
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)
elif fitType=="landau":
    fitFunc = "TMath::Landau(x, [0], [1])"
    nparams = 2
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(100, 200)
elif fitType=="landau_linear":
    fitFunc = "TMath::Landau(x, [0], [1]) + [2] + [3] * x"
    nparams = 4
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(600, 500)
elif fitType=="landau_quadratic":
    fitFunc = "TMath::Landau(x, [0], [1]) + [2] + [3] * x + [4] * x * x"
    nparams = 5
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(600, 500)
elif fitType=="landau_cubic":
    fitFunc = "TMath::Landau(x, [0], [1]) + [2] + [3] * x + [4] * x * x + [5] * x * x * x"
    nparams = 6
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(600, 500)
else:
    print("fitFunc not defined. Please specify.")
    exit()

#skewFit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)
#skewFit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001, 0.0000000001)
#skewFit.SetParameters(5, 500, 500, 50, 0.00001, 0.000001, 0.000000001)

def fit_and_plot(hist, plotname):
    c = TCanvas("")
    latex = TLatex()
    latex.SetNDC()
    
    hist.Scale(1/hist.Integral())
    hist.Fit(fit, "E")
    hist.Draw()
        
    chi2ndof = fit.GetChisquare() / fit.GetNDF()
    latex.DrawText(0.6, 0.2, f'chi2/ndof = {round(chi2ndof,2)}')
    
    c.SaveAs(plotname)
    print(f'Saved plot to {plotname}.')

    
histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")

# store parameters in dictionaries
params = {"tgt":{},
          "pre":{}
          }

if parser.parse_args().doValidation:
    hist_abcdnn = histFile.Get(f'Bprime_mass_ABCDnn_val')
    fit_and_plot(hist_abcdnn, f'{plotDir}/fit_ABCDnn_validation.png')
else:
    for region in ["A", "B", "C", "D", "X", "Y"]:
        params["tgt"][region] = {}
        params["pre"][region] = {}

    # fit
    # hist range: [400, 2500]
    for region in ["A", "B", "C", "D", "X", "Y"]:
        for htype in ["tgt", "pre"]:
            if htype == "tgt":
                hist = histFile.Get(f'Bprime_mass_tgt_{region}_{case}') - histFile.Get(f'Bprime_mass_mnr_{region}_{case}')
            elif htype == "pre":
                hist = histFile.Get(f'Bprime_mass_pre_{region}_{case}')
            else:
                print(f'Undefined histogram type {htype}. Check for typo.')

            fit_and_plot(hist, f'{plotDir}/fit_{htype}_{region}.png')
            
            for i in range(nparams):
                params[htype][region][str(i)] = fit.GetParameter(i)

    histFile.Close()

    # compare
    params_uncert = {}
    for i in ["0", "1", "2", "3", "4", "5", "6"]:
        uncert = 0
        for region in ["A", "B", "C"]: # excluded X, Y. Y p6 oddly different.
            uncert += abs((params["pre"][region][i]-params["tgt"][region][i])/params["tgt"][region][i])

        params_uncert[i] = uncert/3
        print(f'Avg deviation in param{i}: {100*uncert/3}%')



