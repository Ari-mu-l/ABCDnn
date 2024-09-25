import numpy as np
from ROOT import *


#skewFit = TF1("landauFit", "landau", 500, 2500, 2) # consider normalizing the hist. if cannot converge
landauPoly = "TMath::Landau(x, [0], [1]) + [2] + [3] * x + [4] * x * x"
nparams = 3
skewFit = TF1("landauFit", landauPoly, 500, 2500, nparams)

#skewFit.SetParameters(600, 500)

#histFile = TFile.Open("hists_ABCDnn.root", "READ")
histFile = TFile.Open("hists_ABCDnn_500to2500_51.root", "READ")

# store parameters in dictionaries
params = {"tgt":{},
          "pre":{}
          }
lastbin = {"tgt":{},
           "pre":{}
           }

for region in ["A", "B", "C", "D", "X", "Y"]:
    params["tgt"][region] = {}
    params["pre"][region] = {}
#    for i in range(7): # the number of params
#        params["tgt"][region][str(i)] = {}
#        params["pre"][region][str(i)] = {}

# fit
# hist range: [500, 2500]
for region in ["A", "B", "C", "D", "X", "Y"]:
    for htype in ["tgt", "pre"]:
        c = TCanvas("")
        latex = TLatex()
        latex.SetNDC()
        if htype == "tgt":
            hist = histFile.Get(f'Bprime_mass_tgt_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
        elif htype == "pre":
            hist = histFile.Get(f'Bprime_mass_pre_{region}')
        else:
            print(f'Undefined histogram type {htype}. Check for typo.')

            
        # last bin is not fitted. get bin content
        lastbin[htype][region] = hist.GetBinContent(len(hist))
        
        hist.Scale(1/hist.Integral())

        hist.Fit(skewFit, "E")
        hist.Draw()
        
        chi2ndof = skewFit.GetChisquare() / skewFit.GetNDF()
        latex.DrawText(0.5, 0.7, f'chi2/ndof = {round(chi2ndof,2)}')
        
        c.SaveAs(f'fit_plots/fit_{htype}_{region}.png')
        print(f'Saved plot to fit_plots/fit_{htype}_{region}.png')

        for i in range(nparams): # number of parameters
            params[htype][region][str(i)] = skewFit.GetParameter(i)

histFile.Close()
exit()

# compare
params_uncert = {}
for i in ["0", "1", "2", "3", "4", "5", "6"]:
    uncert = 0
    for region in ["A", "B", "C"]: # excluded X, Y. Y p6 oddly different.
        uncert += abs((params["pre"][region][i]-params["tgt"][region][i])/params["tgt"][region][i])
    
    params_uncert[i] = uncert/3
    print(f'Avg deviation in param{i}: {100*uncert/3}%')

lastbin_uncert = 0
for region in ["A", "B", "C"]:
    lastbin_uncert = abs(lastbin["pre"][region]-lastbin["tgt"][region])/lastbin["tgt"][region]
lastbin_uncert = lastbin_uncert/3

# fill last bin separately


# shift
#for i in ["0", "1", "2", "3", "4", "5", "6"]:
    
#    params["pre"]["D"][i] * (1+params_uncert[i])
#    params["pre"]["D"][i] * (1-params_uncert[i])
    #print(f'Resulting param{i}_up is {params["pre"]["D"]"')

# fill last bin separately
