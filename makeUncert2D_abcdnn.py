import ROOT
import os
import json

ROOT.TH2.SetDefaultSumw2(True) # Question: is this needed?
ROOT.gROOT.SetBatch(True) # suppress histogram display
ROOT.gStyle.SetOptStat(0) # no stat box

# histogram settings
bin_lo_BpM = 400 #0
bin_hi_BpM = 2500
bin_lo_ST = 0
bin_hi_ST = 1500
Nbins_BpM = 420 # 2100
Nbins_ST  = 30
validationCut = 850

rebin = 5
Nbins_BpM_actual = int(Nbins_BpM/rebin)

plotDir ='2D_plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

STTag = {"fullST": "",
         "lowST": "V",
         "highST": "2"}

STMap = {"fullST": "D",
         "lowST": "V",
         "highST": "D2"}

def modifyOverflow2D(hist):
    ## Julie comment: ROOT counts bins from 1, I think we need to be using 1 in place of 0 here
    # lower right corner
    # hist.SetBinContent(Nbins_BpM_actual, 0, hist.GetBinContent(Nbins_BpM_actual, 0)+hist.GetBinContent(Nbins_BpM_actual+1, 0))
    # hist.SetBinContent(Nbins_BpM_actual+1, 0, 0)

    # # upper right corner
    # hist.SetBinContent(Nbins_BpM_actual, Nbins_ST, hist.GetBinContent(Nbins_BpM_actual, Nbins_ST)+hist.GetBinContent(Nbins_BpM_actual+1, Nbins_ST)+hist.GetBinContent(Nbins_BpM_actual, Nbins_ST+1)+hist.GetBinContent(Nbins_BpM_actual+1, Nbins_ST+1))
    # hist.SetBinContent(Nbins_BpM_actual+1, Nbins_ST, 0)
    # hist.SetBinContent(Nbins_BpM_actual, Nbins_ST+1, 0)
    # hist.SetBinContent(Nbins_BpM_actual+1, Nbins_ST+1, 0)

    ### Julie comment: I think we actually need to do the entire right EDGE and the entire top EDGE...
    # top edge should be [imass,Nbins_ST] as the bin number, where imass runs 1 through Nbins_BpM_actual
    for imass in range(1,Nbins_BpM_actual+1):
        newtotal = hist.GetBinContent(imass,Nbins_ST)+hist.GetBinContent(imass,Nbins_ST+1)
        hist.SetBinContent(imass,Nbins_ST,newtotal)
        hist.SetBinContent(imass,Nbins_ST+1,0)
    # right edge should be [Nbins_BpM_actual,ist], where ist runs 1 through Nbins_ST
    for ist in range(1,Nbins_ST+1):
        newtotal = hist.GetBinContent(Nbins_BpM_actual,ist)+hist.GetBinContent(Nbins_BpM_actual+1,ist)
        hist.SetBinContent(Nbins_BpM_actual,ist,newtotal)
        hist.SetBinContent(Nbins_BpM_actual+1,ist,0)
        


def getNormalizedTgtPreHists(histFile, histTag):
    hist_tgt = histFile.Get(f'BpMST_dat_{histTag}').Clone()
    hist_mnr = histFile.Get(f'BpMST_mnr_{histTag}').Clone()
    hist_pre = histFile.Get(f'BpMST_pre_{histTag}').Clone()

    hist_tgt.RebinX(rebin)
    hist_mnr.RebinX(rebin)
    hist_pre.RebinX(rebin)
    
    # tgt = dat - mnr
    hist_tgt.Add(hist_mnr, -1.0)

    modifyOverflow2D(hist_tgt)
    modifyOverflow2D(hist_pre)

    hist_tgt.Scale(1/hist_tgt.Integral())
    hist_pre.Scale(1/hist_pre.Integral())

    return hist_tgt, hist_pre


alphaFactors = {}
with open("alphaRatio_factors.json","r") as alphaFile:
    alphaFactors = json.load(alphaFile)
    
def getAlphaRatioTgtPreHists(histFile, histTag):
    _, hist_pre = getNormalizedTgtPreHists(histFile, histTag)

    hist_tgt = histFile.Get(f'BpMST_dat_{histTag}').Clone()
    hist_mnr = histFile.Get(f'BpMST_mnr_{histTag}').Clone()

    hist_tgt.RebinX(rebin)
    hist_mnr.RebinX(rebin)

    hist_tgt.Add(hist_mnr, -1.0)

    modifyOverflow2D(hist_tgt)
    #modifyOverflow2D(hist_pre)

    #TEMP: remove after the naming convention is changed in getAlphaRatio
    if histTag=="D2":
        region = "highST"
    else:
        region = histTag
    
    yield_pred = alphaFactors[case][region]["prediction"]
    hist_pre.Scale(yield_pred)

    return hist_tgt, hist_pre
    

def plotHists2D_All():
    histFile1 = ROOT.TFile.Open(f'hists_ABCDnn_case1_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')
    histFile2 = ROOT.TFile.Open(f'hists_ABCDnn_case2_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')
    histFile3 = ROOT.TFile.Open(f'hists_ABCDnn_case3_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')
    histFile4 = ROOT.TFile.Open(f'hists_ABCDnn_case4_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')

    for region in ["D", "V", "D2"]:
        hist_tgt1, hist_pre1 = getAlphaRatioTgtPreHists(histFile1, f'{region}')
        hist_tgt2, hist_pre2 = getAlphaRatioTgtPreHists(histFile2, f'{region}')
        hist_tgt3, hist_pre3 = getAlphaRatioTgtPreHists(histFile3, f'{region}')
        hist_tgt4, hist_pre4 = getAlphaRatioTgtPreHists(histFile4, f'{region}')

        hist_pre1.Add(hist_pre2)
        hist_pre1.Add(hist_pre3)
        hist_pre1.Add(hist_pre4)
        
        c1 = ROOT.TCanvas(f'c1_{region}', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 900, 600)
        hist_pre1.SetTitle(f'ST_ABCDnn vs BpM_ABCDnn in region {region}')
        hist_pre1.GetXaxis().SetTitle('B mass (GeV)')
        hist_pre1.GetYaxis().SetTitle('ST (GeV)')
        hist_pre1.GetXaxis().SetRangeUser(400,2500) # set viewing range
        hist_pre1.GetYaxis().SetRangeUser(400,1500)
        hist_pre1.Draw("COLZ")
        c1.SaveAs(f'{plotDir}BpMST_ABCDnn_{region}_all.png')
        c1.Close()

    histFile1.Close()
    histFile2.Close()
    histFile3.Close()
    histFile4.Close()
        

def plotHists2D_Separate(case):
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')
    
    for region in ["D", "V", "D2"]:
        hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}')

        # plot prediction hist for V,D,D2
        c1 = ROOT.TCanvas(f'c1_{region}', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 900, 600)
        
        hist_pre.SetTitle(f'ST_ABCDnn vs BpM_ABCDnn in region {region}')
        hist_pre.GetXaxis().SetTitle('B mass (GeV)')
        hist_pre.GetYaxis().SetTitle('ST (GeV)')
        hist_pre.Draw("COLZ")
        c1.SaveAs(f'{plotDir}BpMST_ABCDnn_{region}_{case}.png')
        c1.Close()
  
        # blind for D and highST case1 and 2
        blind = False
        if region=='D' or region=='D2':
            if case=='case1' or case=='case2':
                blind = True

        if not blind:
            # plot target hist for non-blinded cases in V,D,VhighST
            c2 = ROOT.TCanvas(f'c2_{region}', f'ST_target vs BpM_target in region {region}', 900, 600)
            hist_tgt.SetTitle(f'ST_target vs BpM_target in region {region}')
            hist_tgt.GetXaxis().SetTitle('B mass (GeV)')
            hist_tgt.GetYaxis().SetTitle('ST (GeV)')
            hist_tgt.Draw("COLZ")
            c2.SaveAs(f'{plotDir}BpMST_target_{region}_{case}.png')
            c2.Close()

    # plot training uncertainty derived from full ST, low ST, highST
    for STrange in ['fullST','lowST','highST']:
        c3 = ROOT.TCanvas(f'c3_{case}_{STrange}', f'Percentage training uncertainty from {STrange} ({case})', 900, 600)
        hist_trainUncert = histFile.Get(f'BpMST_trainUncert{STrange}')
        hist_trainUncert.SetTitle(f'Percentage training uncertainty from {STrange} ({case})')
        hist_trainUncert.GetXaxis().SetTitle('B mass (GeV)')
        hist_trainUncert.GetYaxis().SetTitle('ST (GeV)')
        hist_trainUncert.Draw("COLZ")
        c3.SaveAs(f'{plotDir}BpMST_trainUncertPercent{STrange}_{case}.png')
        c3.Close()

    # plot 2D correction maps
    for STrange in ["fullST"]:
        c4 = ROOT.TCanvas(f'c4_{case}_{STrange}', f'Percentage correction from {STrange} ({case})', 900, 600)
        hist_Correct = histFile.Get(f'BpMST_Correct{STMap[STrange]}')
        hist_Correct.SetTitle(f'Percentage correction from {STMap[STrange]} ({case})')
        hist_Correct.GetXaxis().SetTitle('B mass (GeV)')
        hist_Correct.GetYaxis().SetTitle('ST (GeV)')
        hist_Correct.GetYaxis().SetRangeUser(400,1500)
        hist_Correct.GetZaxis().SetRangeUser(-1.0,2.0)
        hist_Correct.Draw("COLZ")
        c4.SaveAs(f'{plotDir}BpMST_correctionPercent{STMap[STrange]}_{case}.png')
        c4.Close()

    # plot corrected region D BpM (1D)
    for	STrange in ["fullST"]:
        c5 = ROOT.TCanvas(f'c5_{case}_{STrange}', f'Bprime_mass_ABCDnn corrected with 2D ({case})', 600, 600)
        hist_lowSTCorrected = histFile.Get(f'Bprime_mass_pre_D_withCorrect{STMap[STrange]}').Clone()
        hist_lowSTCorrectedUp = histFile.Get(f'Bprime_mass_pre_D_withCorrect{STMap[STrange]}Up').Clone()
        hist_lowSTCorrectedDn = histFile.Get(f'Bprime_mass_pre_D_withCorrect{STMap[STrange]}Dn').Clone()
        hist_lowSTCorrected.Scale(alphaFactors[case]["D"]["prediction"]/hist_lowSTCorrected.Integral())
        hist_lowSTCorrected.SetTitle(f'Bprime_mass_ABCDnn corrected with {STrange} 2D map ({case})') # fix lowST label
        hist_lowSTCorrected.GetXaxis().SetTitle('B mass (GeV)')
        #hist_lowSTCorrected.GetYaxis().SetTitle('')
        hist_lowSTCorrected.Draw("HIST")
        #hist_lowSTCorrectedUp.Draw("HIST SAME")
        #hist_lowSTCorrectedDn.Draw("HIST SAME")
        c5.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_correctedfromD_{case}.png')
        c5.Close()

    # plot 1D correction derived from 2D
    for STrange in ["fullST"]:
        c6 = ROOT.TCanvas(f'c6_{case}_{STrange}', 'Bprime_mass_ABCDnn correction from 2D ({case})', 600, 600)
        hist1D = histFile.Get(f'Bprime_mass_pre_Correct{STMap[STrange]}')
        hist1D.SetTitle(f'Bprime_mass_ABCDnn correction from {STrange} 2D map ({case})')
        hist1D.GetXaxis().SetTitle('B mass (GeV)')
        hist1D.Draw("HIST")
        c6.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_1Dcorrectfrom2D_{case}.png')
        c6.Close()
        
    
for case in ["case1", "case2", "case3", "case4"]:
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'UPDATE')
    ###################
    # training uncert #
    ###################
    # use A,B,C to calculate train uncert with fullST, lowST,highST
    for STrange in ["fullST","lowST","highST"]:
        hist_trainUncert = ROOT.TH2D(f'BpMST_trainUncert{STrange}_{case}', "BpM_vs_ST", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST, bin_lo_ST, bin_hi_ST)
        for region in ["A", "B", "C"]:
            hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}{STTag[STrange]}')
            
            # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
            hist_tgt.Add(hist_pre, -1.0)
            hist_tgt.Divide(hist_pre)

            # take absolute value abs(PercentageDiff) for training uncert
            # allow negative values for V and highST corrections
            for i in range(Nbins_BpM_actual+1):
                for j in range(Nbins_ST+1):
                    if hist_tgt.GetBinContent(i,j)<0:
                        hist_tgt.SetBinContent(i,j,-hist_tgt.GetBinContent(i,j))
            # add contribution to training uncert
            hist_trainUncert.Add(hist_tgt, 1.0)
        # average over A,B,C
        hist_trainUncert.Scale(1/3)
        hist_trainUncert.Write(f'BpMST_trainUncert{STrange}')
        print(f'Saved BpMST_trainUncert{STrange} to {case}')

    ##############
    # Correction #
    ##############
    #for region in ["V", "D2", "D"]: # D2 is the highST part of region D
    for region in ["V", "D"]:    
        hist_Correction = ROOT.TH2D(f'BpMST_Correct{region}_{case}', "BpM_vs_ST", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST, bin_lo_ST, bin_hi_ST) # TEMP: rebin 5
        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFile, f'{region}')

        for i in range(Nbins_BpM_actual+1):
            for j in range(Nbins_ST+1):
                if hist_tgt.GetBinContent(i,j)<=10:
                    hist_Correction.SetBinContent(i,j,0)
        
        # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
        hist_tgt.Add(hist_pre, -1.0)
        hist_tgt.Divide(hist_pre)
        
        hist_Correction.Add(hist_tgt, 1.0)

        print(f'Correction in bin (20,10): {hist_Correction.GetBinContent(20,10)}')
        #exit()
        if "D" in region: # only keep derivation from case3,4 in region V
            # assgin the same correction for case2 and 3
            if case=="case3":
                hist_Correction.Write(f'BpMST_Correct{region}')
                print(f'Saved BpMST_Correct{region} to {case}')
                histFilePartner = ROOT.TFile.Open(f'hists_ABCDnn_case2_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'UPDATE')
                hist_Correction.Write(f'BpMST_Correct{region}')
                print(f'Saved BpMST_Correct{region} to case2')
                histFilePartner.Close()
            # assign the same correction for case1 and 4
            if case=="case4":
                hist_Correction.Write(f'BpMST_Correct{region}')
                print(f'Saved BpMST_Correct{region} to {case}')
                histFilePartner = ROOT.TFile.Open(f'hists_ABCDnn_case1_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'UPDATE')
                hist_Correction.Write(f'BpMST_Correct{region}')
                print(f'Saved BpMST_Correct{region} to case1')
                histFilePartner.Close()
        else: # write correction for case1,2,3,4 derived from region V
            hist_Correction.Write(f'BpMST_Correct{region}')
            print(f'Saved BpMST_Correct{region} to {case}')

    histFile.Close()

# apply correction
def applyCorrection(corrType):
    for case in ["case1", "case2", "case3", "case4"]:
        histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'UPDATE')

        _, hist_pre = getAlphaRatioTgtPreHists(histFile, 'D')
        
        hist_preUp = hist_pre.Clone('preUp')
        hist_preDn = hist_pre.Clone('preDn')
        hist_cor = histFile.Get(f'BpMST_Correct{corrType}').Clone('cor')

        # note: case1 and 2 only apply to low bkg region
        # for i in range(Nbins_BpM):
        #     for j in range(Nbins_ST):
        #         print(hist_cor.GetYaxis().FindFixBin(850))
                
    
        # pre = pre + pre*correction
        hist_cor.Multiply(hist_pre)
        hist_pre.Add(hist_cor)

        # preUp = pre + pre*2*correction
        hist_preUp.Add(hist_cor)
        hist_preUp.Add(hist_cor)

        # preDn = pre + pre*0, so do nothing 

        hist_pre_1DUp = hist_preUp.ProjectionX()
        hist_pre_1DDn = hist_preDn.ProjectionX()
        hist_pre_1D = hist_pre.ProjectionX()

        hist_cor1D = hist_pre_1D.Clone()
        hist_pre_1D.Print()
        hist_pre_1DDn.Print()
        hist_cor1D.Add(hist_pre_1DDn, -1.0) # corrected - original
        hist_cor1D.Print()
        hist_cor1D.Divide(hist_pre_1DDn)
        hist_cor1D.Print()

        hist_pre_1D.Write(f'Bprime_mass_pre_D_withCorrect{corrType}')
        hist_pre_1DUp.Write(f'Bprime_mass_pre_D_withCorrect{corrType}Up')
        hist_pre_1DDn.Write(f'Bprime_mass_pre_D_withCorrect{corrType}Dn')
        hist_cor1D.Write(f'Bprime_mass_pre_Correct{corrType}')

        histFile.Close()

applyCorrection('V')
#applyCorrection('D2')
applyCorrection('D')


# plot histograms
for case in ["case1", "case2", "case3", "case4"]:
    plotHists2D_Separate(case)

plotHists2D_All()
