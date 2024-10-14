import os
import numpy as np
from ROOT import *
import json

gStyle.SetOptFit(1)
gStyle.SetOptStat(0)
gROOT.SetBatch(True) # suppress histogram display

# define the fit function
# f(x) = 2 pdf(x) * cdf(a x)
# f(x) = 2/w pdf((x-xi)/2) * cdf(a*(x-xi)/2)
# f(x) = f(x) + ax^2 + bx + c

binlo = 400
binhi = 2500
bins = 42 #43

# store parameters in dictionaries
params = {"case14":{"tgt":{},"pre":{},"val":{}},
          "case23":{"tgt":{},"pre":{},"val":{}},
          "case1":{"tgt":{},"pre":{},"val":{}},
          "case2":{"tgt":{},"pre":{},"val":{}},
          "case3":{"tgt":{},"pre":{},"val":{}},
          "case4":{"tgt":{},"pre":{},"val":{}}
          }

previousbins = {}

lastbin = {"case14":{"tgt":{},"pre":{},"val":None},
           "case23":{"tgt":{},"pre":{},"val":None},
           "case1":{"tgt":{},"pre":{},"val":None},
           "case2":{"tgt":{},"pre":{},"val":None},
           "case3":{"tgt":{},"pre":{},"val":None},
           "case4":{"tgt":{},"pre":{},"val":None}
          }

pred_uncert = {"Description":"Prediction uncertainty",
               "case14":{},
               "case23":{},
               "case1":{},
               "case2":{},
               "case3":{},
               "case4":{}
               }

#fit_apply = {}

tag = {"case14": "allWlep",
       "case23": "allTlep",
       "case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

hist_D = {"case14":{"ABCDnn":None, "fit":None, "target":None},
          "case23":{"ABCDnn":None, "fit":None, "target":None},
          "case1":{"ABCDnn":None, "fit":None, "target":None},
          "case2":{"ABCDnn":None, "fit":None, "target":None},
          "case3":{"ABCDnn":None, "fit":None, "target":None},
          "case4":{"ABCDnn":None, "fit":None, "target":None}
          }

hist_validation = {"case14":{"ABCDnn":None, "fit":None, "target":None},
                   "case23":{"ABCDnn":None, "fit":None, "target":None},
                   "case1":{"ABCDnn":None, "fit":None, "target":None},
                   "case2":{"ABCDnn":None, "fit":None, "target":None},
                   "case3":{"ABCDnn":None, "fit":None, "target":None},
                   "case4":{"ABCDnn":None, "fit":None, "target":None}
                   }

yield_ABCDnn_D = {}
yield_ABCDnn_val = {}
yield_target_D = {}

#######
# Fit #
#######

# skewNorm_4 with initialization 1 worked the best for both case14 and case 23
fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x + [8] * x * x * x * x"
nparams=9


def fit_and_plot(hist, plotname):
    fit = TF1(f'fitFunc', fitFunc,400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)
    
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

    return fit

def fitHist(case):
    print(f'Fitting hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root...')

    plotDir = f'fit_plots/{case}_{binlo}to{binhi}_{bins}'
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")

    for region in ["A", "B", "C", "D", "X", "Y"]:
        params[case]["tgt"][region] = {}
        params[case]["pre"][region] = {}

    # fit
    # hist range: [400, 2500]
    for region in ["A", "B", "C", "D", "X", "Y"]:
        for htype in ["tgt", "pre"]:
            if htype == "tgt":
                hist = histFile.Get(f'Bprime_mass_tgt_{region}_{case}') - histFile.Get(f'Bprime_mass_mnr_{region}_{case}')
                if region=="D":
                    hist_D[case]["target"] = hist.Clone() # store normalized
                    yield_target_D[case] = hist.Integral()
                    hist_D[case]["target"].Scale(1/hist.Integral())
                    hist_D[case]["target"].SetDirectory(0)
                    # TODO: check underflow bin
                    #hist.Print("all")
                    #exit()
            elif htype == "pre":
                hist = histFile.Get(f'Bprime_mass_pre_{region}_{case}')
                if region=="D":
                    hist_D[case]["ABCDnn"] = hist.Clone() # store normalized
                    yield_ABCDnn_D[case] = hist.GetEntries()
                    hist_D[case]["ABCDnn"].Scale(1/hist.Integral())
                    hist_D[case]["ABCDnn"].SetDirectory(0)
            else:
                print(f'Undefined histogram type {htype}. Check for typo.')

            fit = fit_and_plot(hist, f'{plotDir}/fit_{htype}_{region}.png') # normalizes hist

            # last bin is not fitted. get bin content. after scaling. capture shape only
            lastbin[case][htype][region] = hist.GetBinContent(bins+1)
            for i in range(nparams):
                params[case][htype][region][f'param{str(i)}'] = [fit.GetParameter(i), fit.GetParError(i)]

            if region=="D" and htype=="pre": # store the fit to create histogram
                hist_D[case]["fit"] = hist.Clone()
                hist_D[case]["fit"].SetDirectory(0)
                
    # fit validation
    hist_val = histFile.Get(f'Bprime_mass_ABCDnn_val')
    yield_ABCDnn_val[case] = hist_val.GetEntries()
    hist_val.Scale(1/hist_val.Integral())
    hist_validation[case]["ABCDnn"] = hist_val # store normalized
    hist_validation[case]["ABCDnn"].SetDirectory(0)
    lastbin[case]['val'] = hist_val.GetBinContent(bins+1)

    fit_val = fit_and_plot(hist_val, f'{plotDir}/fit_ABCDnn_validation.png')
    
    for i in range(nparams):
        params[case]['val'][f'param{str(i)}'] = [fit_val.GetParameter(i), fit_val.GetParError(i)]

    hist_validation[case]["fit"] = hist_val.Clone()
    hist_validation[case]["fit"].SetDirectory(0)
    #hist_validation[case]["ABCDnn"] = hist_val
    #hist_validation[case]["ABCDnn"].SetDirectory(0)
    hist_validation[case]["target"] = histFile.Get(f'Bprime_mass_data_val') - histFile.Get(f'Bprime_mass_minor_val') # store normalized
    hist_validation[case]["target"].Scale(hist_validation[case]["target"].Integral())
    hist_validation[case]["target"].SetDirectory(0)
    
    histFile.Close()
    
    # compare
    for i in range(nparams):
        uncert = 0
        for region in ["A", "B", "C"]: # excluded X, Y
            uncert += abs((params[case]["pre"][region][f'param{i}'][0]-params[case]["tgt"][region][f'param{i}'][0])/params[case]["tgt"][region][f'param{i}'][0]) # params[case]["pre"][region][i] is a list of [param, err]
        pred_uncert[case][f'param{i}'] = uncert/3
        print(f'Avg deviation in param{i}: {100*uncert/3}%')

    lastbin_uncert=0
    for region in ["A", "B", "C"]:
        lastbin_uncert += abs(lastbin[case]["pre"][region]-lastbin[case]["tgt"][region])/lastbin[case]["tgt"][region]
    pred_uncert[case]["lastbin"] = lastbin_uncert/3

fitHist("case14")
fitHist("case23")
fitHist("case1")
fitHist("case2")
fitHist("case3")
fitHist("case4")

# save parameters and last bin to a json file
json_obj = json.dumps(params, indent=4)
with open("fit_parameters.json", "w") as outjsonfile:
    outjsonfile.write(json_obj)
print("Saved parameter and last bin info to fit_parameters.json")

json_obj = json.dumps(pred_uncert, indent=4)
with open("pred_uncert.json", "w") as outjsonfile:
    outjsonfile.write(json_obj)
print("Saved parameter and last bin info to pred_uncert.json")

#############################
# create histogram from fit #
#############################
alphaFactors = {}
with open("alphaRatio_factors.json","r") as alphaFile:
    alphaFactors = json.load(alphaFile)
counts = {}
with open("counts.json","r") as countsFile:
    counts = json.load(countsFile)
    
if not os.path.exists('application_plots'):
    os.makedirs('application_plots')

def shapePlot(case, fit, hist_gen, plotname):
    c1 = TCanvas("c1", "c1")
    legend = TLegend(0.5,0.2,0.9,0.3)

    hist_gen.SetTitle(f'Shape verification {case}')
    hist_gen.SetLineColor(kBlue)
    hist_gen.Draw()

    fit.SetLineColor(kRed)
    fit.Draw("SAME")
    
    legend.AddEntry(hist_gen, "Histogram from the fit", "l")
    legend.AddEntry(fit, "Histogram directly from ABCDnn", "l")
    legend.Draw()

    c1.SaveAs(plotname)
    c1.Close()
    
def targetAgreementPlot(case, fit, hist_gen, hist_target, hist_abcdnn, doValidation):
    c2 = TCanvas("c2", "c2", 800, 800)
    pad1 = TPad("hist_plot", "hist_plot", 0.05, 0.3, 1, 1)
    pad1.SetBottomMargin(0) #join upper and lower plot
    pad1.SetLeftMargin(0.1)
    pad1.Draw()
    pad1.cd()
    
    legend = TLegend(0.6,0.6,0.9,0.9)

    hist_gen.SetLineColor(kRed)
    hist_gen.Draw("HIST")
    hist_target.SetLineColor(kBlack)
    hist_target.Draw("SAME")
    hist_abcdnn.SetLineColor(kBlue)
    hist_abcdnn.Draw("HIST SAME")
    
    legend.AddEntry(hist_gen, "Histogram from fit", "l")
    legend.AddEntry(hist_abcdnn, "Directly from ABCDnn", "l")
    legend.AddEntry(hist_target, "Data-minor", "l")
    legend.Draw()    
    
    c2.cd()
    pad2 = TPad("ratio_plot", "ratio_plot", 0.05, 0.05, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.2)
    pad2.SetLeftMargin(0.1)
    pad2.SetGrid()
    pad2.Draw()
    pad2.cd()

    line = TF1("line", "1", binlo, binhi, 0)
    
    hist_ratio = hist_target / hist_gen
    hist_ratio.SetTitle("")
    #hist_ratio.GetYaxis().SetRangeUser(0.5,1.5) #TEMP
    hist_ratio.GetYaxis().SetRangeUser(0,2)
    hist_ratio.GetYaxis().SetTitle("fit/target")

    hist_ratio.SetMarkerStyle(20)
    hist_ratio.SetLineColor(kBlack)
    hist_ratio.Draw("pex0")
    line.SetLineColor(kBlack)
    line.Draw("SAME")
    #hist_gen.Print("all")
    #hist_trueABCDnn[case].Print("all")

    if doValidation:
        hist_gen.SetTitle(f'Major background in validation region {case}')
    else:
        hist_gen.SetTitle(f'Major background {case}')
    hist_gen.GetYaxis().SetTitle("Events/50GeV")
    hist_gen.GetYaxis().SetTitleSize(20)
    hist_gen.GetYaxis().SetTitleFont(43)

    hist_ratio.GetYaxis().SetTitleSize(20)
    hist_ratio.GetYaxis().SetTitleFont(43)

    #text = TText(0.4, 0.4,"Statistical uncertainty only")
    #text.Draw()

    if doValidation:
        plotname = f'application_plots/GeneratedHist_with_target_{case}_validation.png'
    else:
        plotname = f'application_plots/GeneratedHist_with_target_{case}.png'
    c2.SaveAs(plotname)
    c2.Close()
    
def createHist(case, doValidation):
    
    #hist_pred.Scale(1/hist_pred.Integral())
    #hist_pred.SetLineColor(kBlue)
    #hist_pred.Draw()
    #fit_apply[case].Draw()

    fit = hist_D[case]["fit"].GetFunction("fitFunc").Clone()
    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins) # TEMP
    hist_gen = fit.CreateHistogram()
    
    shapePlot(case, fit, hist_gen, f'application_plots/GeneratedHist_with_fit_{case}.png')

    # scale and plot with target and ABCDnn histograms
    # TODO: histogram error bars
    # TODO: finness scaling
    #hist_pred.Scale(hist_D[case]["ABCDnn"].Integral(1,bins))
    #hist_pred.SetBinContent(bins+1, lastbin[case]["pre"]["D"])
    #print(hist_D[case]["ABCDnn"].Integral(bins-1,bins))
    #exit()

    # TODO: test from now. Add validation counting in plotValidation.py
    if doValidation:
        hist_gen.SetBinContent(bins, hist_gen.GetBinContent(bins)+lastbin[case]["val"])
        #hist_gen.Scale(alphaFactors[case]["factor"] * yield_ABCDnn_val[case])
        hist_gen.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
        hist_target = hist_validation[case]["target"] # normalized to 1
        hist_abcdnn = hist_validation[case]["ABCDnn"] # normalized to 1
        hist_abcdnn.SetBinContent(bins, hist_abcdnn.GetBinContent(bins)+lastbin[case]["pre"]["D"]) 
        #hist_abcdnn.Scale(alphaFactors[case]["factor"] * yield_ABCDnn_val[case])
        hist_target.Scale(counts[case]["val"]["data"]-counts[case]["val"]["minor"])
        hist_abcdnn.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
    else:
        hist_gen.SetBinContent(bins, hist_gen.GetBinContent(bins)+lastbin[case]["pre"]["D"])
        #hist_gen.Scale(alphaFactors[case]["factor"] * yield_ABCDnn_D[case]) # normalized to 1
        hist_gen.Scale(alphaFactors[case]["prediction"])
        hist_target = hist_D[case]["target"] # normalized to 1
        hist_target.Scale(counts[case]["D"]["data"]-counts[case]["D"]["minor"])
        hist_abcdnn = hist_D[case]["ABCDnn"] # normalized to 1
        hist_abcdnn.SetBinContent(bins, hist_abcdnn.GetBinContent(bins)+lastbin[case]["pre"]["D"]) 
        #hist_abcdnn.Scale(alphaFactors[case]["factor"] * yield_ABCDnn_D[case])
        hist_abcdnn.Scale(alphaFactors[case]["prediction"])
    
    #for i in range(1,bins):
    #   hist_target.SetBinError(i, np.sqrt(hist_target.GetBinContent(i)))
    for i in range(bins+1):
       hist_gen.SetBinError(i, np.sqrt(hist_gen.GetBinContent(i)))
    for i in range(bins+1):
       hist_abcdnn.SetBinError(i, np.sqrt(hist_abcdnn.GetBinContent(i)))    

    if (case=="case3") or (case=="case4") or doValidation:
        hist_target.SetBinContent(bins, hist_target.GetBinContent(bins)+hist_target.GetBinContent(bins+1))
        #hist_target.SetBInerror-(bins, np.sqrt(hist_target.GetBinContent(bins)))
    else:
        for i in range(bins):
            if hist_target.GetBinCenter(i) > 1000:
                hist_target.SetBinContent(i,0)
            else:
                hist_target.SetBinError(i, np.sqrt(hist_target.GetBinContent(i)))
        #hist_target.SetBinContent(bins+1, 0)

    #for i in range(bins):
    #    hist_target.SetBinError(i, np.sqrt(hist_target.GetBinContent(i)))

    #print(case, hist_target.Integral(), hist_abcdnn.Integral())
    #exit()
    targetAgreementPlot(case, fit, hist_gen, hist_target, hist_abcdnn, doValidation)

    if doValidation:
        hist_gen.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_val__major')
    else:
        hist_gen.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_D__major')
    
outFile = TFile.Open("templates_BpMass_ABCDnn_138fbfb.root","RECREATE")
createHist("case14", True)
createHist("case23", True)
createHist("case1", True)
createHist("case2", True)
createHist("case3", True)
createHist("case4", True)
createHist("case14", False)
createHist("case23", False)
createHist("case1", False)
createHist("case2", False)
createHist("case3", False)
createHist("case4", False)
outFile.Close()


# plot validation
#def plotValidation(case):
#    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
#    hist_data  = histFile.Get("Bprime_mass_data_val")
#    hist_minor = histFile.Get("Bprime_mass_minor_val")

    # TODO: how to get validation ABCDnn. easiest: fit on validation ABCDnn
#    tempFile = TFile.Open("templates_BpMass_ABCDnn_138fbfb.root","RECREATE")
    #hist_fit = tempFile.Get(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_D__major')

    #hist_model = hist_minor + hist_fit


# shift
def shiftLastBin(case, htype, shift):
    fit = hist_D[case]["fit"].GetFunction("fitFunc").Clone()
    #fit.SetRange(0,2500)
    fit.SetNpx(bins)
    #fit.SetNpx(50)

    hist_bin = fit.CreateHistogram()

    if htype=="val":
        if shift=="Up":
            hist_bin.SetBinContent(bins, lastbin[case][htype]*(1+pred_uncert[case]["lastbin"]))
        else:
            hist_bin.SetBinContent(bins, lastbin[case][htype]*(1-pred_uncert[case]["lastbin"]))
        
        hist_bin.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
    else:
        if shift=="Up":
            hist_bin.SetBinContent(bins, lastbin[case][htype]["D"]*(1+pred_uncert[case]["lastbin"]))
        else:
            hist_bin.SetBinContent(bins, lastbin[case][htype]["D"]*(1-pred_uncert[case]["lastbin"]))
            
        hist_bin.Scale(alphaFactors[case]["prediction"])

    hist_bin.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_D__major__lastbin{shift}')
        
def shiftParam(case, htype, i, shift):
    #print(case, htype, i)
    if htype=="val":
        fit = hist_validation[case]["fit"].GetFunction("fitFunc").Clone()
        if shift=="Up":
            # todo: add fit uncertainty into pred_uncert
            fit.SetParameter(i, params[case][htype][f'param{i}'][0]*(1+pred_uncert[case][f'param{i}']))
        else:
            fit.SetParameter(i, params[case][htype][f'param{i}'][0]*(1-pred_uncert[case][f'param{i}']))
    else:
        fit = hist_D[case]["fit"].GetFunction("fitFunc").Clone()
        if shift=="Up":
            # todo: add fit uncertainty into pred_uncert
            fit.SetParameter(i, params[case][htype]["D"][f'param{i}'][0]*(1+pred_uncert[case][f'param{i}']))
        else:
            fit.SetParameter(i, params[case][htype]["D"][f'param{i}'][0]*(1-pred_uncert[case][f'param{i}']))
    
    #fit.SetParameter(i, shiftedparam)
    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins)
    hist_param = fit.CreateHistogram()

    if htype=="val":
        hist_param.SetBinContent(bins, lastbin[case][htype])
        hist_param.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
    else:
        hist_param.SetBinContent(bins, lastbin[case][htype]["D"])
        hist_param.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
    
    hist_param.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_D__major__param{i}{shift}')

    #fit.SetParameter(i, params[case][htype][f'param{i}'][0]) # check if the fit reverts back

    
outFile = TFile.Open("templates_BpMass_ABCDnn_138fbfb.root","UPDATE")
for case in ["case14", "case23", "case1", "case2", "case3", "case4"]:
    for shift in ["Up", "Down"]:
        shiftLastBin(case, "val", shift)
        shiftLastBin(case, "pre", shift)

        for i in range(nparams):
            shiftParam(case, "val", i, shift)
            shiftParam(case, "pre", i, shift)

outFile.Close()

# todo: add factor uncertainty
