# copy the desired rootfiles to working directory first

import os
import numpy as np
from ROOT import *
import json

gStyle.SetOptFit(1)
gStyle.SetOptStat(0)
gROOT.SetBatch(True) # suppress histogram display
TH1.SetDefaultSumw2(True)

binlo = 400
binhi = 2500
bins = 2100

doV2 = False
withFit = False
separateUncertCases = True

# store parameters in dictionaries
# region: A, B, C, D, X, Y
params = {"case1":{"tgt":{},"pre":{}},
          "case2":{"tgt":{},"pre":{}},
          "case3":{"tgt":{},"pre":{}},
          "case4":{"tgt":{},"pre":{}},
          } 
pred_uncert = {"Description":"Prediction uncertainty",
               "case1":{},
               "case2":{},
               "case3":{},
               "case4":{}
               }

# region: "D", "V"
normalization = {
                 "case1":{},
                 "case2":{},
                 "case3":{},
                 "case4":{}
                 }

tag = {"case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

outDir = '/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates'

def modifyOverflow(hist, bins):
    hist.SetBinContent(bins, hist.GetBinContent(bins)+hist.GetBinContent(bins+1))
    hist.SetBinContent(bins+1, 0)
    
#### TRAINING UNCERTAINTY BY BINS ####
def getPredUncert(case):
    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
    
    hist_dev = TH1D(f'train_uncert_{case}', f'Training Uncertainty ({case})', bins, binlo, binhi)
    for region in ["A", "B", "C"]:
        hist_pre = histFile.Get(f'Bprime_mass_pre_{region}')
        hist_tgt = histFile.Get(f'Bprime_mass_dat_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
        hist_pre.Scale(1/hist_pre.Integral())
        hist_tgt.Scale(1/hist_tgt.Integral())
        modifyOverflow(hist_pre,bins)
        modifyOverflow(hist_tgt,bins)
        hist_dev += (hist_tgt - hist_pre)/hist_pre
    hist_dev.Scale(1/3) # average over 3 regions
    
    uncertFile.cd()
    hist_dev.Write()
    histFile.Close()

uncertFile = TFile.Open(f'hists_trainUncert_{binlo}to{binhi}_{bins}_pNet.root', "RECREATE")
#getPredUncert("case14")
#getPredUncert("case23")
getPredUncert("case1")
getPredUncert("case2")
getPredUncert("case3")
getPredUncert("case4")
uncertFile.Close()

#############################
# create histogram with corr#
#############################

alphaFactors = {}
with open("alphaRatio_factors.json","r") as alphaFile:
    alphaFactors = json.load(alphaFile)
counts = {}
with open("counts.json","r") as countsFile:
    counts = json.load(countsFile)
    
def shapePlot(region, case, hist_gen, hist_ABCDnn, step):
    c1 = TCanvas("c1", "c1")
    legend = TLegend(0.5,0.2,0.9,0.3)
    
    hist_gen.SetTitle(f'Shape verification {case}')
    hist_gen.SetLineColor(kBlue)
    #hist_gen.Scale(1/hist_gen.Integral())
    hist_gen.Draw("HIST")

    hist_ABCDnn.SetLineColor(kBlack)
    hist_ABCDnn.Scale(1/hist_ABCDnn.Integral())
    hist_ABCDnn.Draw("SAME")
    
    legend.AddEntry(hist_gen, "Histogram from the fit", "l")
    legend.AddEntry(hist_ABCDnn, "Histogram directly from ABCDnn", "l")
    #legend.AddEntry(fit, "Fit function from ABCDnn", "l")
    legend.Draw()

    #hist_ratio = hist_ABCDnn/hist_gen
    #hist_ratio.Print("all")

    c1.SaveAs(f'application_plots/GeneratedHist_with_fit_{case}_{region}_{step}.png')
    c1.Close()
    
def targetAgreementPlot(region, case, fit, hist_gen, hist_target, hist_abcdnn):
    c2 = TCanvas("c2", "c2", 800, 800)
    pad1 = TPad("hist_plot", "hist_plot", 0.05, 0.3, 1, 1)
    pad1.SetBottomMargin(0.01) #join upper and lower plot
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
    # TEMP: turned of negative bin adjustment
    # Need to turn back on if not using smoothing (smoothing script takes care of negative bins)
    #for i in range(bins+1): # deals with negative bins
    #    if hist.GetBinContent(i)<0:
    #        hist.SetBinContent(i,0)
    return hist
    
def createHist(case, region):
    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
    hist_original = histFile.Get(f'Bprime_mass_pre_{region}').Clone() #with pNetSF

    normalization[case][region] = alphaFactors[case][region]["prediction"]/hist_original.Integral()
    hist_original.Scale(normalization[case][region])
    modifyOverflow(hist_original,bins)
        
    hist_out = fillHistogram(hist_original)
    
    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major')
                if case=="case1":
                    hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetTtagUp')
                    hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetTtagDown')
                elif case=="case2":
                    hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetWtagUp')
                    hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetWtagDown')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major')
            if case=="case1":
                hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetTtagUp')
                hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetTtagDn')
            elif case=="case2":
                hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetWtagUp')
                hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetWtagDn')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")
    
for case in ["case1", "case2", "case3", "case4"]:
    createHist(case, "D") #TEMP
    #createHist(case, "D2") #TEMP
    createHist(case, "V") #TEMP
    # if doV2:
    #     createHist(case, "D")
    #     createHist(case, "V")
    # else:
    #     createHist(case, "D")

# shift
def shiftTrainingUncert(case, region, shift):
    if withFit:
        fit = TF1(f'fitFunc', fitFunc, binlo,binhi, nparams)
        for i in range(nparams):
            fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])
        fit.SetNpx(bins)
        hist_bin = fit.CreateHistogram() # fit method
        hist_bin.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"]) #fit method
    else:
        histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
        # if case=='case1' or case=="case4":
        #     histFile = TFile.Open(f'logBpMlogST_mmd1_case14_random113/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ") #TEMP: test other models
        # else:
        #     histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
        
        hist_bin = histFile.Get(f'Bprime_mass_pre_{region}').Clone()
    
    modifyOverflow(hist_bin,bins) # take overflow bin and add to last bin. set over flow bin to 0

    uncertFile = TFile.Open(f'hists_trainUncert_{binlo}to{binhi}_{bins}_pNet.root', 'READ')
    if separateUncertCases:
        uncertHist = uncertFile.Get(f'train_uncert_{case}') # apply separete uncerts
    else:
        if case=="case1" or case=="case4": # apply combined uncerts
            uncertHist = uncertFile.Get("train_uncert_case14")
        else: # assumes not applying to combined cases (case14 and 23)
            uncertHist = uncertFile.Get("train_uncert_case23")

    if shift=="Up":
        for i in range(bins+1):
            hist_bin.SetBinContent(i, hist_bin.GetBinContent(i) * (1 + uncertHist.GetBinContent(i)))
    else:
        for i in range(bins+1):
    	    hist_bin.SetBinContent(i, hist_bin.GetBinContent(i) * (1 - uncertHist.GetBinContent(i)))

    hist_bin.Scale(normalization[case][region])
    
    hist_out = fillHistogram(hist_bin)

    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__train{shift}')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__train{shift}')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major__train{shift}')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")
    
for case in ["case1", "case2", "case3", "case4"]:
    for shift in ["Up", "Down"]:
        shiftTrainingUncert(case, "D", shift) #TEMP
        #shiftTrainingUncert(case, "D2", shift) #TEMP
        ##shiftFactor(case, "D", shift)
        ##shiftLastBin(case, "D", shift) # not needed for bin train uncert method
        
        shiftTrainingUncert(case, "V", shift) #TEMP
        ##shiftFactor(case, "V", shift)
        ##shiftLastBin(case, "V", shift)
        # if doV2:
        #     shiftLastBin(case, "V", shift)
        #     shiftLastBin(case, "D", shift)
        #     shiftFactor(case, "V", shift)
        #     shiftFactor(case, "D", shift)
        # else:
        #     shiftLastBin(case, "D", shift)
        #     shiftFactor(case, "D", shift)
