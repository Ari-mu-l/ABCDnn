import os
import numpy as np
from ROOT import *
import json

gStyle.SetOptFit(1)
gROOT.SetBatch(True) # suppress histogram display

# define the fit function
# f(x) = 2 pdf(x) * cdf(a x)
# f(x) = 2/w pdf((x-xi)/2) * cdf(a*(x-xi)/2)
# f(x) = f(x) + ax^2 + bx + c

binlo = 400
binhi = 2500
bins = 43 #43

# store parameters in dictionaries
params = {"case14":{"tgt":{},"pre":{}},
          "case23":{"tgt":{},"pre":{}},
          }

previousbins = {}

lastbin = {"case14":{"tgt":{},"pre":{}},
           "case23":{"tgt":{},"pre":{}},
          }

pred_uncert = {"Description":"Prediction uncertainty","case14":{}, "case23":{}}

#fit_apply = {}

tag = {"case14": "allWlep",
       "case23": "allTlep",
       "case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

hist_D= {"case14":{},
         "case23":{}}

# skewNorm_cubic with initialization 1 worked the best for both case14 and case 23
fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x"
nparams=8

def fitHist(case):
    print(f'Fitting hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root...')

    plotDir = f'fit_plots/{case}_{binlo}to{binhi}_{bins}'
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    fit = TF1(f'fitFunc', fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)

    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")

    for region in ["A", "B", "C", "D", "X", "Y"]:
        params[case]["tgt"][region] = {}
        params[case]["pre"][region] = {}
    #    for i in range(nparams): # the number of params
    #        params[case]["tgt"][region][str(i)] = {}
    #        params[case]["pre"][region][str(i)] = {}

    # fit
    # hist range: [400, 2500]
    for region in ["A", "B", "C", "D", "X", "Y"]:
        for htype in ["tgt", "pre"]:
            c = TCanvas("")
            latex = TLatex()
            latex.SetNDC()
            if htype == "tgt":
                hist = histFile.Get(f'Bprime_mass_tgt_{region}_{case}') - histFile.Get(f'Bprime_mass_mnr_{region}_{case}')
                if region=="D":
                    hist_D[case]["Target"] = hist.Clone()
                    hist_D[case]["Target"].Scale(1/hist.Integral())
                    hist_D[case]["Target"].SetDirectory(0)
                    # TODO: check underflow bin
                    #hist.Print("all")
                    #exit()
            elif htype == "pre":
                hist = histFile.Get(f'Bprime_mass_pre_{region}_{case}')
                if region=="D":
                    hist_D[case]["ABCDnn"] = hist.Clone()
                    hist_D[case]["ABCDnn"].Scale(1/hist.Integral())
                    hist_D[case]["ABCDnn"].SetDirectory(0)
            else:
                print(f'Undefined histogram type {htype}. Check for typo.')

            hist.Scale(1/hist.Integral())
            
            # last bin is not fitted. get bin content. after scaling. capture shape only
            lastbin[case][htype][region] = hist.GetBinContent(len(hist))
            
            hist.Fit(fit, "E")
            hist.Draw()

            chi2ndof = fit.GetChisquare() / fit.GetNDF()
            latex.DrawText(0.6, 0.2, f'chi2/ndof = {round(chi2ndof,2)}')

            c.SaveAs(f'{plotDir}/fit_{htype}_{region}.png')
            print(f'Saved plot to {plotDir}/fit_{htype}_{region}.png')

            for i in range(nparams):
                params[case][htype][region][f'param{str(i)}'] = [fit.GetParameter(i), fit.GetParError(i)]

            if region=="D" and htype=="pre": # store the fit to create histogram
                hist_D[case]["Fitted"] = hist.Clone()
                hist_D[case]["Fitted"].SetDirectory(0)
                
    
    histFile.Close()
    
    # compare
    for i in ["param0", "param1", "param2", "param3", "param4", "param5", "param6", "param7"]:
        uncert = 0
        for region in ["A", "B", "C"]: # excluded X, Y
            uncert += abs((params[case]["pre"][region][i][0]-params[case]["tgt"][region][i][0])/params[case]["tgt"][region][i][0]) # params[case]["pre"][region][i] is a list of [param, err]
        pred_uncert[case][i] = uncert/3
        print(f'Avg deviation in {i}: {100*uncert/3}%')

    lastbin_uncert=0
    for region in ["A", "B", "C"]:
        lastbin_uncert += abs(lastbin[case]["pre"][region]-lastbin[case]["tgt"][region])/lastbin[case]["tgt"][region]
    pred_uncert[case]["lastbin"] = lastbin_uncert/3

fitHist("case14")
fitHist("case23")

# save parameters and last bin to a json file
json_obj = json.dumps(params, indent=4)
with open("fit_parameters.json", "w") as outjsonfile:
    outjsonfile.write(json_obj)
print("Saved parameter and last bin info to fit_parameters.json")

json_obj = json.dumps(pred_uncert, indent=4)
with open("pred_uncert.json", "w") as outjsonfile:
    outjsonfile.write(json_obj)
print("Saved parameter and last bin info to pred_uncert.json")

# create histogram from fit func
alphaFactors = {}
with open("alphaRatio_factors.json","r") as alphaFile:
    alphaFactors = json.load(alphaFile)
    
if not os.path.exists('application_plots'):
    os.makedirs('application_plots')
    
def createHist(case):
    #fit_apply[case].SetNpx(bins)
    #hist_pred = fit_apply[case].GetHistogram() #hist_pred = fit_apply["case14"].CreateHistogram().Rebin(2)

    c = TCanvas("")
    legend = TLegend(0.5,0.2,0.9,0.3)
    
    #hist_pred.Scale(1/hist_pred.Integral())
    #hist_pred.SetLineColor(kBlue)
    #hist_pred.Draw()
    #fit_apply[case].Draw()

    fit = hist_D[case]["Fitted"].GetFunction("fitFunc")
    fit.SetNpx(bins)
    hist_pred = fit.CreateHistogram()
    hist_pred.SetTitle("")
    hist_pred.SetLineColor(kBlue)
    hist_pred.Draw()

    hist_D[case]["Fitted"].SetLineColor(kBlack)
    hist_D[case]["Fitted"].Draw("SAME")
    
    legend.AddEntry(hist_pred, "Histogram from the fit", "l")
    legend.AddEntry(hist_D[case]["Fitted"], "Histogram directly from ABCDnn", "l")
    legend.Draw()

    c.SaveAs(f'application_plots/GeneratedHist_with_fit_{case}.png')

    # TODO: histogram error bars
    # TODO: finness scaling
    #hist_pred.Scale(hist_D[case]["ABCDnn"].Integral(1,bins))
    #hist_pred.SetBinContent(bins+1, lastbin[case]["pre"]["D"])
    #print(hist_D[case]["ABCDnn"].Integral(bins-1,bins))
    #exit()
    legend = TLegend(0.6,0.6,0.9,0.9)
    
    hist_pred.Scale(alphaFactors[case]["prediction"])
    hist_pred.Draw("HIST")

    hist_D[case]["Target"].Scale(alphaFactors[case]["prediction"])
    hist_D[case]["Target"].Draw("HIST SAME")

    hist_D[case]["ABCDnn"].Scale(alphaFactors[case]["prediction"])
    hist_D[case]["ABCDnn"].Draw("HIST SAME")
    
    legend.AddEntry(hist_pred, "Histogram from fit", "l")
    legend.AddEntry(hist_D[case]["ABCDnn"], "Directly from ABCDnn", "l")
    legend.AddEntry(hist_D[case]["Target"], "Data-minor", "l")
    legend.Draw()
    
    c.SaveAs(f'application_plots/GeneratedHist_with_target_{case}.png')

    hist_pred.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_D__major')
    #hist_pred.Print("all")
    #hist_trueABCDnn[case].Print("all")

    
histFile = TFile.Open("templates_BpMass_ABCDnn_138fbfb.root","RECREATE")
createHist("case14")
createHist("case23")
histFile.Close()
exit()


# fill last bin separately


# shift
#for i in ["0", "1", "2", "3", "4", "5", "6"]:
    
#    params["pre"]["D"][i] * (1+params_uncert[i])
#    params["pre"]["D"][i] * (1-params_uncert[i])
    #print(f'Resulting param{i}_up is {params["pre"]["D"]"')

# fill last bin separately
