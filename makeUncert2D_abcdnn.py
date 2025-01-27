import ROOT
import os

ROOT.TH2.SetDefaultSumw2(True) # Question: is this needed?
ROOT.gROOT.SetBatch(True) # suppress histogram display

# histogram settings
bin_lo_BpM = 400 #0
bin_hi_BpM = 2500
bin_lo_ST = 0
bin_hi_ST = 1500
Nbins_BpM = 420 # 2100
Nbins_ST  = 30
validationCut = 850

plotDir ='2D_plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

def modifyOverflow2D(hist):
    # upper left corner
    hist.SetBinContent(Nbins_BpM, 0, hist.GetBinContent(Nbins_BpM, 0)+hist.GetBinContent(Nbins_BpM+1, 0))
    hist.SetBinContent(Nbins_BpM+1, 0, 0)

    # upper right corner
    hist.SetBinContent(Nbins_BpM, Nbins_ST, hist.GetBinContent(Nbins_BpM, Nbins_ST)+hist.GetBinContent(Nbins_BpM+1, Nbins_ST)+hist.GetBinContent(Nbins_BpM, Nbins_ST+1)+hist.GetBinContent(Nbins_BpM+1, Nbins_ST+1))
    hist.SetBinContent(Nbins_BpM+1, Nbins_ST, 0)
    hist.SetBinContent(Nbins_BpM, Nbins_ST+1, 0)
    hist.SetBinContent(Nbins_BpM+1, Nbins_ST+1, 0)


def getNormalizedTgtPreHists(histFile, histTag):
    hist_tgt = histFile.Get(f'BpMST_dat_{histTag}').Clone()
    hist_mnr = histFile.Get(f'BpMST_mnr_{histTag}').Clone()
    hist_pre = histFile.Get(f'BpMST_pre_{histTag}').Clone()

    # tgt = dat - mnr
    hist_tgt.Add(hist_mnr, -1.0)

    modifyOverflow2D(hist_tgt)
    modifyOverflow2D(hist_pre)

    hist_tgt.Scale(1/hist_tgt.Integral())
    hist_pre.Scale(1/hist_pre.Integral())

    return hist_tgt, hist_pre

def plotHists2D_all():
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')

    

def plotHists2D_Separate(case):
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'READ')
    
    for region in ["D", "V", "D2"]:
        hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}')

        # plot prediction hist for V,D,D2
        c1 = ROOT.TCanvas(f'c1_{region}', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 400, 400)
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
            c2 = ROOT.TCanvas(f'c2_{region}', f'ST_target vs BpM_target in region {region}', 400, 400)
            hist_tgt.SetTitle(f'ST_target vs BpM_target in region {region}')
            hist_tgt.GetXaxis().SetTitle('B mass (GeV)')
            hist_tgt.GetYaxis().SetTitle('ST (GeV)')
            hist_tgt.Draw("COLZ")
            c2.SaveAs(f'{plotDir}BpMST_target_{region}_{case}.png')
            c2.Close()

    # plot training uncertainty derived from full ST
    c3 = ROOT.TCanvas(f'c3_{case}', f'Percentage training uncertainty from full ST ({case})', 400, 400)
    hist_trainUncertFullST = histFile.Get('BpMST_trainUncert')
    hist_trainUncertFullST.SetTitle(f'Percentage training uncertainty from full ST ({case})')
    hist_trainUncertFullST.GetXaxis().SetTitle('B mass (GeV)')
    hist_trainUncertFullST.GetYaxis().SetTitle('ST (GeV)')
    hist_trainUncertFullST.Draw("COLZ")
    c3.SaveAs(f'{plotDir}BpMST_trainUncertPercentFullST_{case}.png')
    c3.Close()

    # plot training uncertainty derived from low ST
    c4 = ROOT.TCanvas(f'c4_{case}', f'Percentage training uncertainty from low ST ({case})', 400, 400)
    hist_trainUncertLowST = histFile.Get('BpMST_trainUncertlowST')
    hist_trainUncertLowST.SetTitle(f'Percentage training uncertainty from low ST ({case})')
    hist_trainUncertLowST.GetXaxis().SetTitle('B mass (GeV)')
    hist_trainUncertLowST.GetYaxis().SetTitle('ST (GeV)')
    hist_trainUncertLowST.Draw("COLZ")
    c4.SaveAs(f'{plotDir}BpMST_trainUncertPercentLowST_{case}.png')
    c4.Close()

    # plot training uncertainty derived from high ST
    c5 = ROOT.TCanvas(f'c5_{case}', f'Percentage training uncertainty from high ST ({case})', 400, 400)
    hist_trainUncertHighST = histFile.Get('BpMST_trainUncerthighST')
    hist_trainUncertHighST.SetTitle(f'Percentage training uncertainty from high ST ({case})')
    hist_trainUncertHighST.GetXaxis().SetTitle('B mass (GeV)')
    hist_trainUncertHighST.GetYaxis().SetTitle('ST (GeV)')
    hist_trainUncertHighST.Draw("COLZ")
    c5.SaveAs(f'{plotDir}BpMST_trainUncertPercentHighST_{case}.png')
    c5.Close()

    # plot VR correction
    c6 = ROOT.TCanvas(f'c6_{case}', f'Percentage correction from V ({case})', 400, 400)
    hist_valCorrect = histFile.Get('BpMST_CorrectV')
    hist_valCorrect.SetTitle(f'Percentage correction from V ({case})')
    hist_valCorrect.GetXaxis().SetTitle('B mass (GeV)')
    hist_valCorrect.GetYaxis().SetTitle('ST (GeV)')
    hist_valCorrect.Draw("COLZ")
    c6.SaveAs(f'{plotDir}BpMST_correctionPercentV_{case}.png')
    c6.Close()
    
    # plot highST correction
    c7 = ROOT.TCanvas(f'c7_{case}', 'Percentage correction from highST ({case})', 400, 400)
    hist_highSTCorrect= histFile.Get('BpMST_CorrecthighST')
    hist_highSTCorrect.SetTitle('Percentage correction from highST ({case})')
    hist_highSTCorrect.GetXaxis().SetTitle('B mass (GeV)')
    hist_highSTCorrect.GetYaxis().SetTitle('ST (GeV)')
    hist_highSTCorrect.Draw("COLZ")
    c7.SaveAs(f'{plotDir}BpMST_correctionPercentHighST_{case}.png')
    c7.Close()

    histFile.Close()

STTag = {"fullST": "",
         "lowST": "V",
         "highST": "2"}
    
for case in ["case1", "case2", "case3", "case4"]:
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet.root', 'UPDATE')
    ###################
    # training uncert #
    ###################
    # use A,B,C to calculate train uncert with fullST, lowST,highST
    for STrange in ["fullST","lowST","highST"]:
        hist_trainUncert = ROOT.TH2D(f'BpMST_trainUncert{STrange}_{case}', "BpM_vs_ST", Nbins_BpM, bin_lo_BpM, bin_hi_BpM, Nbins_ST, bin_lo_ST, bin_hi_ST)
        for region in ["A", "B", "C"]:
            hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}{STTag[STrange]}')
            
            # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
            hist_tgt.Add(hist_pre, -1.0)
            hist_tgt.Divide(hist_pre)

            # take absolute value abs(PercentageDiff) for training uncert
            # allow negative values for V and highST corrections
            for i in range(Nbins_BpM+1):
                for j in range(Nbins_ST+1):
                    if hist_tgt.GetBinContent(i,j)<0:
                        hist_tgt.SetBinContent(i,j,-hist_tgt.GetBinContent(i,j))
            # add contribution to training uncert
            hist_trainUncert.Add(hist_tgt, 1.0)
        # average over A,B,C
        hist_trainUncert.Scale(1/3)
        hist_trainUncert.Write(f'BpMST_trainUncert{STrange}')
        print(f'Saved BpMST_trainUncert{STrange} to {case}')
            
    for region in ["V", "D2"]: # D2 is the highST part of region D
        hist_Correction = ROOT.TH2D(f'BpMST_Correct{region}_{case}', "BpM_vs_ST", Nbins_BpM, bin_lo_BpM, bin_hi_BpM, Nbins_ST, bin_lo_ST, bin_hi_ST)
        hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}')
        
        # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
        hist_tgt.Add(hist_pre, -1.0)
        hist_tgt.Divide(hist_pre)
        
        hist_Correction.Add(hist_tgt, 1.0)
        
        if region=="D2": # only keep derivation from case3,4 in region V
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

# plot histograms
for case in ["case1", "case2", "case3", "case4"]:
    plotHists2D(case)
