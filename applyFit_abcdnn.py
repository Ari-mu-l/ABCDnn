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
bins = 42

# store parameters in dictionaries
# region: A, B, C, D, X, Y
# param: param0, param1, ..., lastbin
params = {"case14":{"tgt":{},"pre":{}},
          "case23":{"tgt":{},"pre":{}},
          "case1":{"tgt":{},"pre":{}},
          "case2":{"tgt":{},"pre":{}},
          "case3":{"tgt":{},"pre":{}},
          "case4":{"tgt":{},"pre":{}},
          } 
pred_uncert = {"Description":"Prediction uncertainty",
               "case14":{},
               "case23":{},
               "case1":{},
               "case2":{},
               "case3":{},
               "case4":{}
               }

# region: "D", "V"
normalization = {"case14":{},
                 "case23":{},
                 "case1":{},
                 "case2":{},
                 "case3":{},
                 "case4":{}
                 }

tag = {"case14": "allWlep",
       "case23": "allTlep",
       "case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

#name_map = {"A": "A", "B": "B", "C": "C", "D": "D", "X": "X", "Y": "Y", "V": "val"}

outDir = '/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates'
#######
# Fit #
#######

# skewNorm_4 with initialization 1 worked the best for both case14 and case 23
# skewNorm_cubic
#fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x + [8] * x * x * x * x"
#nparams=9

# skewNorm_cubic_2
fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * x + [5] * x * x + [6] * x * x * x"
nparams=7

def fit_and_plot(hist, plotname):
    fit = TF1(f'fitFunc', fitFunc,400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.000001, 0.00000001)
    
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

    for region in ["A", "B", "C", "D", "X", "Y", "V"]:
        params[case]["tgt"][region] = {}
        params[case]["pre"][region] = {}

    # fit
    # hist range: [400, 2500]
    for region in ["A", "B", "C", "D", "X", "Y", "V"]:
        for htype in ["tgt", "pre"]:
            if htype == "tgt":
                hist = histFile.Get(f'Bprime_mass_dat_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
            else:
                hist = histFile.Get(f'Bprime_mass_pre_{region}')

            fit = fit_and_plot(hist, f'{plotDir}/fit_{htype}_{region}.png') # normalizes hist

            # last bin is not fitted. get bin content. after scaling. capture shape only
            params[case][htype][region]["lastbin"] = hist.GetBinContent(bins+1)
            for i in range(nparams):
                params[case][htype][region][f'param{str(i)}'] = [fit.GetParameter(i), fit.GetParError(i)]

    histFile.Close()
    
    # compare
    for i in range(nparams):
        uncert = 0
        if i<4:
            for region in ["A", "B", "C"]: # excluded X, Y
                uncert += abs((params[case]["pre"][region][f'param{i}'][0]-params[case]["tgt"][region][f'param{i}'][0])/params[case]["tgt"][region][f'param{i}'][0]) # params[case]["pre"][region][i] is a list of [param, err]
            train_uncert = (uncert/3)*params[case]["pre"]["D"][f'param{i}'][0] # absolute shift for D
            fit_uncer = params[case]["pre"]["D"][f'param{i}'][1]
            pred_uncert[case][f'param{i}'] = abs(np.sqrt(train_uncert**2+fit_uncer**2)/params[case]["pre"]["D"][f'param{i}'][0])
        else:
            pred_uncert[case][f'param{i}'] = abs(params[case]["pre"]["D"][f'param{i}'][1]/params[case]["pre"]["D"][f'param{i}'][0])
        print(f'Uncertainty in param{i}: ',pred_uncert[case][f'param{i}'])

    #for i in range()

    lastbin_uncert=0
    for region in ["A", "B", "C"]:
        lastbin_uncert += abs(params[case]["pre"][region]["lastbin"]-params[case]["tgt"][region]["lastbin"])/params[case]["tgt"][region]["lastbin"]
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

def shapePlot(region, case, hist_gen, hist_ABCDnn, step):
    c1 = TCanvas("c1", "c1")
    legend = TLegend(0.5,0.2,0.9,0.3)
    
    hist_gen.SetTitle(f'Shape verification {case}')
    hist_gen.SetLineColor(kBlue)
    hist_gen.Draw("HIST")

    hist_ABCDnn.SetLineColor(kBlack)
    hist_ABCDnn.Draw("SAME")
    
    legend.AddEntry(hist_gen, "Histogram from the fit", "l")
    legend.AddEntry(hist_ABCDnn, "Histogram directly from ABCDnn", "l")
    #legend.AddEntry(fit, "Fit function from ABCDnn", "l")
    legend.Draw()

    c1.SaveAs(f'application_plots/GeneratedHist_with_fit_{case}_{region}_{step}.png')
    c1.Close()
    
def targetAgreementPlot(region, case, fit, hist_gen, hist_target, hist_abcdnn):
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

    hist_gen.SetTitle(f'Major background {case} in region {region}')
    hist_gen.GetYaxis().SetTitle("Events/50GeV")
    hist_gen.GetYaxis().SetTitleSize(20)
    hist_gen.GetYaxis().SetTitleFont(43)

    hist_ratio.GetYaxis().SetTitleSize(20)
    hist_ratio.GetYaxis().SetTitleFont(43)

    #text = TText(0.4, 0.4,"Statistical uncertainty only")
    #text.Draw()

    c2.SaveAs(f'application_plots/GeneratedHist_with_target_{case}_{region}.png')
    c2.Close()

def fillHistogram(hist):
    hist_new = TH1F("hist_new","hist_new",bins,400,2500)
    for i in range(bins+1):
        hist_new.SetBinContent(i,hist.GetBinContent(i))
        #hist_new.SetBinError(i,np.sqrt(hist.GetBinContent(i)))
    return hist_new
    
    return hist_new
def createHist(case, region):
    fit = TF1(f'fitFunc', fitFunc,400, 2500, nparams)
    for i in range(nparams):
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])
    
    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins)
    hist_gen = fit.CreateHistogram()
    #for j in range(10):
    #    if hist_gen.GetXaxis().GetBinCenter(j) < 400:
    #        hist_gen.SetBinContent(j, 0)
    
    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
    hist_target = histFile.Get(f'Bprime_mass_dat_{region}').Clone(f'Bprime_mass_tgt_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}').Clone()
    hist_abcdnn = histFile.Get(f'Bprime_mass_pre_{region}').Clone()
    hist_abcdnn_shape = hist_abcdnn.Clone()
    #print('before scaling: ', hist_abcdnn_shape.GetBinError(10))
    #hist_abcdnn_shape.Scale(1/hist_abcdnn.Integral())
    #print('after scaling: ', hist_abcdnn_shape.GetBinError(10))
    #exit()
    
    shapePlot(region, case, hist_gen, hist_abcdnn_shape, "step1")

    hist_gen.SetBinContent(bins, hist_gen.GetBinContent(bins)+params[case]["pre"][region]["lastbin"])
    hist_gen.SetBinContent(bins+1, 0) # overflow added to the previous bin
    hist_abcdnn_shape.SetBinContent(bins, hist_abcdnn_shape.GetBinContent(bins)+hist_abcdnn_shape.GetBinContent(bins+1))
    hist_abcdnn_shape.SetBinContent(bins+1, 0)
    hist_abcdnn.SetBinContent(bins, hist_abcdnn.GetBinContent(bins)+hist_abcdnn.GetBinContent(bins+1))
    hist_abcdnn.SetBinContent(bins+1, 0)
    
    shapePlot(region, case, hist_gen, hist_abcdnn_shape, "step2")

    #if region=="V": #TODO: check that this is correct
        #normalization[case][region] = (alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))/(hist_gen.Integral())
    #else:
    #    normalization[case][region] = alphaFactors[case]["prediction"]/(hist_gen.Integral())
    normalization[case][region] = alphaFactors[case][region]["prediction"]/(hist_gen.Integral()) 
    hist_gen.Scale(normalization[case][region])
    hist_abcdnn_shape.Scale(normalization[case][region]) 

    # scale and plot with target and ABCDnn histograms

    #hist_abcdnn.SetBinContent(bins, hist_abcdnn.GetBinContent(bins)+hist_abcdnn.GetBinContent(bins+1))
    #hist_abcdnn.SetBinContent(bins+1, 0)
    #hist_abcdnn.Scale(alphaFactors[case]["factor"])

    #print(hist_gen.Integral(), hist_abcdnn_shape.Integral(), hist_abcdnn.Integral())
    shapePlot(region, case, hist_gen, hist_abcdnn_shape, "step3")

    hist_target.SetBinContent(bins, hist_target.GetBinContent(bins)+hist_target.GetBinContent(bins+1))
    hist_target.SetBinContent(bins+1, 0)
    hist_target.Scale(counts[case][region]["data"]/hist_target.Integral())

    if (case!="case3") and (case!="case4") and (region == "D"):
        for i in range(bins+1):
            if hist_target.GetBinCenter(i) > 1000:
                hist_target.SetBinContent(i,0)

    # for i in range(bins+1):
    #    hist_gen.SetBinError(i, np.sqrt(hist_gen.GetBinContent(i)))
    #    hist_abcdnn.SetBinError(i, np.sqrt(hist_abcdnn.GetBinContent(i)))
    #    hist_target.SetBinError(i, np.sqrt(hist_target.GetBinContent(i)))
       
    targetAgreementPlot(region, case, fit, hist_gen, hist_target, hist_abcdnn_shape)

    hist_out = fillHistogram(hist_gen)
    outFile = TFile.Open(f'{outDir}/templates{region}_Oct2024_42bins/templates_BpMass_ABCDnn_138fbfb.root',"UPDATE")
    hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major')
    outFile.Close()

    
for case in ["case14", "case23", "case1", "case2", "case3", "case4"]:
    for region in ["V", "D"]:
        createHist(case, region)

# shift
def shiftLastBin(case, region, shift):
    fit = TF1(f'fitFunc', fitFunc,400, 2500, nparams)
    for i in range(nparams):
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])

    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins)

    hist_bin = fit.CreateHistogram()
    #for j in range(10):
    #    if hist_bin.GetXaxis().GetBinCenter(j) < 400:
    #        hist_bin.SetBinContent(j, 0)
            
    if shift=="Up":
        hist_bin.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"]*(1+pred_uncert[case]["lastbin"]))
    else:
        hist_bin.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"]*(1-pred_uncert[case]["lastbin"]))

    hist_bin.SetBinContent(bins, hist_bin.GetBinContent(bins)+hist_bin.GetBinContent(bins+1))
    hist_bin.SetBinContent(bins+1, 0)
    #hist_bin.Scale(1/hist_bin.Integral())

    #if region=="val":
        #hist_bin.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
    #else:
        #hist_bin.Scale(alphaFactors[case]["prediction"])

    hist_bin.Scale(normalization[case][region])
        
    # for i in range(bins+1):
    #     hist_bin.SetBinError(i, np.sqrt(hist_bin.GetBinContent(i)))

    outFile = TFile.Open(f'{outDir}/templates{region}_Oct2024_42bins/templates_BpMass_ABCDnn_138fbfb.root',"UPDATE")

    hist_out = fillHistogram(hist_bin)
    #hist_bin.SetTitle(f'{tag[case]}_{name_map[region]}__major__lastbin{shift}')
    hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__lastbin{shift}')
    outFile.Close()
        
def shiftParam(case, region, i, shift):
    fit = TF1(f'fitFunc', fitFunc,400, 2500, nparams)
    for j in range(nparams):
        fit.SetParameter(j, params[case]["pre"][region][f'param{j}'][0])

    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins)

    fit_original = fit.Clone()
    
    if shift=="Up":
        # TODO: add fit uncertainty into pred_uncert
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0]*(1+pred_uncert[case][f'param{i}']))
    else:
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0]*(1-pred_uncert[case][f'param{i}']))

    if case=="case4":
        hist_1 = fit_original.CreateHistogram()
        hist_2 = fit.CreateHistogram()
        
        hist_1.Scale(1/hist_1.Integral())
        hist_2.Scale(1/hist_2.Integral())
        
        c3 = TCanvas("")
        hist_1.SetLineColor(kBlack)
        hist_2.SetLineColor(kRed)
        hist_1.Draw("HIST")
        hist_2.Draw("HIST SAME")
        
        c3.SaveAs(f'application_plots/shapeshift_param{i}{shift}_{region}.png')

    hist_param = fit.CreateHistogram()
    #for j in range(10):
    #    if hist_param.GetXaxis().GetBinCenter(j) < 400:
    #        hist_param.SetBinContent(j, 0)

    hist_param.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"])
    hist_param.SetBinContent(bins, hist_param.GetBinContent(bins)+hist_param.GetBinContent(bins+1))
    hist_param.SetBinContent(bins+1, 0)
    #hist_param.Scale(1/hist_param.Integral())
    
    # if region=="val":
    #     hist_param.Scale(alphaFactors[case]["prediction"] * (counts[case]["val"]["unweighted"]/counts[case]["D"]["unweighted"]))
    # else:
    #     hist_param.Scale(alphaFactors[case]["prediction"])

    hist_param.Scale(normalization[case][region])
    
    # for	j in range(bins+1):
    #     if hist_param.GetBinContent(j)<0:
    #         #print(case, region, f'param{i}', pred_uncert[case][f'param{i}'])
    #         #print(j, hist_param.GetBinContent(j))
    #         #hist_param.Print("all")
    #         #break
    #         hist_param.SetBinContent(j, 0)
    #         hist_param.SetBinError(j, 0)
    #     else:
    #         hist_param.SetBinError(j, np.sqrt(hist_param.GetBinContent(j)))
        
    outFile = TFile.Open(f'{outDir}/templates{region}_Oct2024_42bins/templates_BpMass_ABCDnn_138fbfb.root',"UPDATE")

    hist_out = fillHistogram(hist_param)
    #hist_param.SetTitle(f'{tag[case]}_{name_map[region]}__major__param{i}{shift}')
    hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__param{i}{shift}')
    outFile.Close()

    #fit.SetParameter(i, params[case][htype][f'param{i}'][0]) # check if the fit reverts back

    
#outFile = TFile.Open("templates_BpMass_ABCDnn_138fbfb.root","UPDATE")
for case in ["case1", "case2", "case3", "case4"]:
    for shift in ["Up", "Down"]:
        shiftLastBin(case, "V", shift)
        shiftLastBin(case, "D", shift)

        for i in range(nparams):
            #if i<4: #TEMP skip polynomial for now
            shiftParam(case, "V", i, shift)
            shiftParam(case, "D", i, shift)

#outFile.Close()

# todo: add factor uncertainty
