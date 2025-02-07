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
bin_hi_ST = 1500 #1530 #1500
Nbins_BpM = 420 # 2100
Nbins_ST  = 30 #18 #30
validationCut = 850
statCutoff = 0 #10

unblind_BpM = 700
unblind_ST = 850

rebinX = 2 #10
rebinY = 1 #5
Nbins_BpM_actual = int(Nbins_BpM/rebinX)
Nbins_ST_actual = int(Nbins_ST/rebinY)

year = '_2016' # '', '_2016', '_2016APV'
varyBinSize = True

plotDir ='2D_plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

STTag = {"fullST": "",
         "lowST":  "V",
         "highST": "2"}

STMap = {"fullST": "D",
         "lowST":  "V",
         "highST": "D2"}

# def modifyBinning(hist):
#     # scan across BpM
#     xbinsList = [hist.GetXaxis().GetBinUpEdge(Nbins_BpM_actual)]
#     nEvt = 0
#     i = 0
#     for j in range(Nbins_ST_actual+1):
#         while i<bin_hi_BpM and nEvt<5:
#             if hist.GetBinContent(i,j)>0:
#                 nEvt += hist.GetBinContent(i,j) 
#             i+=1

#         if nEvt<5:
#             print(f'Merging all BpM bins for ST bin = {j} yields: {nEvt}')
                
#     #print(xbinsList)
#     exit()
               

def modifyOverflow2D(hist):
    ## Julie comment: ROOT counts bins from 1, I think we need to be using 1 in place of 0 here
    ## Julie comment: I think we actually need to do the entire right EDGE and the entire top EDGE...
    # top edge should be [imass,Nbins_ST] as the bin number, where imass runs 1 through Nbins_BpM_actual
    for imass in range(1,Nbins_BpM_actual+1):
        newtotal = hist.GetBinContent(imass,Nbins_ST_actual)+hist.GetBinContent(imass,Nbins_ST_actual+1)
        hist.SetBinContent(imass,Nbins_ST_actual,newtotal)
        hist.SetBinContent(imass,Nbins_ST_actual+1,0)
    # right edge should be [Nbins_BpM_actual,ist], where ist runs 1 through Nbins_ST
    for ist in range(1,Nbins_ST_actual+1):
        newtotal = hist.GetBinContent(Nbins_BpM_actual,ist)+hist.GetBinContent(Nbins_BpM_actual+1,ist)
        hist.SetBinContent(Nbins_BpM_actual,ist,newtotal)
        hist.SetBinContent(Nbins_BpM_actual+1,ist,0)
        

def getNormalizedTgtPreHists(histFile, histTag, getTgt=True):
    # TEMP: Named 2D hist of pNetUp and Dn as Bprime_mass
    objectName = 'BpMST'
    if 'pNet' in histTag:
        objectName = 'Bprime_mass'
    #hist_pre = histFile.Get(f'BpMST_pre_{histTag}').Clone(f'BpMST_pre_{histTag}')
    hist_pre = histFile.Get(f'{objectName}_pre_{histTag}').Clone(f'BpMST_pre_{histTag}')
    hist_pre.RebinX(rebinX)
    hist_pre.RebinY(rebinY)
    modifyOverflow2D(hist_pre)
    hist_pre.Scale(1/hist_pre.Integral())
    
    if getTgt:
        hist_tgt = histFile.Get(f'{objectName}_dat_{histTag}').Clone(f'BpMST_dat_{histTag}')
        hist_mnr = histFile.Get(f'{objectName}_mnr_{histTag}').Clone(f'BpMST_mnr_{histTag}')
        hist_tgt.RebinX(rebinX)
        hist_mnr.RebinX(rebinX)
        hist_tgt.RebinY(rebinY)
        hist_mnr.RebinY(rebinY)
        
        # tgt = dat - mnr
        hist_tgt.Add(hist_mnr, -1.0)

        modifyOverflow2D(hist_tgt)
        hist_tgt.Scale(1/hist_tgt.Integral())

        return hist_tgt, hist_pre
    else:
        return hist_pre


alphaFactors = {}
with open(f'alphaRatio_factors{year}.json',"r") as alphaFile:
    alphaFactors = json.load(alphaFile)
    
def getAlphaRatioTgtPreHists(histFile, histTag, case, getTgt=True):
    #TEMP: remove after the naming convention is changed in getAlphaRatio
    if histTag=="D2":
        region = "highST"
    else:
        region = histTag
        
    # TEMP: Named 2D hist of pNetUp and Dn as Bprime_mass
    objectName = 'BpMST'
    if 'pNet' in histTag:
        objectName = 'Bprime_mass'
        region = histTag[0]
        
    if getTgt:
        _, hist_pre = getNormalizedTgtPreHists(histFile, histTag, getTgt)
    else:
        hist_pre = getNormalizedTgtPreHists(histFile, histTag, getTgt)

    yield_pred = alphaFactors[case][region]["prediction"]
    hist_pre.Scale(yield_pred)
    
    if getTgt:
        hist_tgt = histFile.Get(f'{objectName}_dat_{histTag}').Clone(f'BpMST_dat_{histTag}')
        hist_mnr = histFile.Get(f'{objectName}_mnr_{histTag}').Clone(f'BpMST_mnr_{histTag}')
        hist_tgt.RebinX(rebinX)
        hist_mnr.RebinX(rebinX)
        hist_tgt.RebinY(rebinY)
        hist_mnr.RebinY(rebinY)
        hist_tgt.Add(hist_mnr, -1.0)
        modifyOverflow2D(hist_tgt)
                    
        return hist_tgt, hist_pre
    else:
        return hist_pre
    

def plotHists2D_All():
    histFile1 = ROOT.TFile.Open(f'hists_ABCDnn_case1_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
    histFile2 = ROOT.TFile.Open(f'hists_ABCDnn_case2_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
    histFile3 = ROOT.TFile.Open(f'hists_ABCDnn_case3_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
    histFile4 = ROOT.TFile.Open(f'hists_ABCDnn_case4_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')

    for region in ["D", "V", "D2"]:
        hist_tgt1, hist_pre1 = getAlphaRatioTgtPreHists(histFile1, f'{region}', 'case1')
        hist_tgt2, hist_pre2 = getAlphaRatioTgtPreHists(histFile2, f'{region}', 'case2')
        hist_tgt3, hist_pre3 = getAlphaRatioTgtPreHists(histFile3, f'{region}', 'case3')
        hist_tgt4, hist_pre4 = getAlphaRatioTgtPreHists(histFile4, f'{region}', 'case4')

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
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
    
    for region in ["D", "V", "D2"]:
        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFile, f'{region}', case)

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
        if year=='' and (region=='D' or region=='D2'):
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
    #for STrange in ['fullST','lowST','highST']:
    for STrange in ['fullST']:
        c3 = ROOT.TCanvas(f'c3_{case}_{STrange}', f'Percentage training uncertainty from {STrange} ({case})', 900, 600)
        hist_trainUncert = histFile.Get(f'BpMST_trainUncert{STrange}')
        hist_trainUncert.SetTitle(f'Percentage training uncertainty from {STrange} ({case})')
        hist_trainUncert.GetXaxis().SetTitle('B mass (GeV)')
        hist_trainUncert.GetYaxis().SetTitle('ST (GeV)')
        hist_trainUncert.GetYaxis().SetRangeUser(400,1500)
        hist_trainUncert.GetZaxis().SetRangeUser(-1.0,2.0)
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
        hist1D = histFile.Get(f'Bprime_mass_pre_Correct{STMap[STrange]}onD')
        hist1D.SetTitle(f'Bprime_mass_ABCDnn correction from {STrange} 2D map ({case})')
        hist1D.GetXaxis().SetTitle('B mass (GeV)')
        hist1D.Draw("HIST")
        c6.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_1Dcorrectfrom2D_{case}.png')
        c6.Close()
        
def addHistograms():
    for case in ["case1", "case2", "case3", "case4"]:
        histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'UPDATE')
        ###################
        # training uncert #
        ###################
        # use A,B,C to calculate train uncert with fullST, lowST,highST
        for STrange in ["fullST"]: #,"lowST","highST"]:
            
            # Alternative 1:
            for region in ["A", "B", "C"]:
                hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}{STTag[STrange]}')
                
                #modifyBinning(hist_tgt)

                # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
                hist_tgt.Add(hist_pre, -1.0)
                hist_tgt.Divide(hist_pre)

                # take absolute value abs(PercentageDiff) for training uncert
                # add contribution to training uncert
                #for i in range(Nbins_BpM_actual+1):
                #    for j in range(Nbins_ST_actual+1):
                #        hist_tgt.SetBinError(i,j,0) # set bin error to 0, so that it acts as a pure scale factor
                #        if hist_tgt.GetBinContent(i,j)<0:
                #            hist_tgt.SetBinContent(i,j,-hist_tgt.GetBinContent(i,j))
     
            # average over A,B,C
            hist_tgt.Scale(1/3)
            hist_tgt.Write(f'BpMST_trainUncert{STrange}')
            print(f'Saved BpMST_trainUncert{STrange} to {case}')
            
            # # Alternative 2:
            # # weighted average
            # hist_tgtABC = ROOT.TH2D(f'BpMST_tgt{STrange}_{case}', "BpM_vs_ST", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST_actual, bin_lo_ST, bin_hi_ST)
            # hist_preABC = ROOT.TH2D(f'BpMST_pre{STrange}_{case}', "BpM_vs_ST", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST_actual, bin_lo_ST, bin_hi_ST)
            # for region in ["A", "B", "C"]:
            #     hist_tgt = histFile.Get(f'BpMST_dat_{region}').Clone(f'dat_{region}')
            #     hist_mnr = histFile.Get(f'BpMST_mnr_{region}').Clone(f'mnr_{region}')
            #     hist_pre = histFile.Get(f'BpMST_pre_{region}').Clone(f'pre_{region}')
                
            #     hist_tgt.Add(hist_mnr, -1.0)

            #     hist_tgt.RebinX(rebinX)
            #     hist_pre.RebinX(rebinX)

            #     hist_tgt.RebinY(rebinY)
            #     hist_pre.RebinY(rebinY)
                
            #     hist_tgtABC.Add(hist_tgt)
            #     hist_preABC.Add(hist_pre)

            # modifyOverflow2D(hist_tgtABC)
            # modifyOverflow2D(hist_preABC)

            # hist_tgtABC.Scale(1/hist_tgtABC.Integral())
            # hist_preABC.Scale(1/hist_preABC.Integral())

            # # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
            # hist_tgtABC.Add(hist_preABC, -1.0)
            # hist_tgtABC.Divide(hist_preABC)

            # for i in range(Nbins_BpM_actual+1):
            #     for j in range(Nbins_ST_actual+1):
            #         hist_tgtABC.SetBinError(i,j,0) # set bin error to 0, so that it acts as a pure scale factor

            # hist_tgtABC.Write(f'BpMST_trainUncert{STrange}')
            # print(f'Saved BpMST_trainUncert{STrange} to {case}')

        ##############
        # Correction #
        ##############
        #for region in ["V", "D2", "D"]: # D2 is the highST part of region D. not tested
        for region in ["V", "D"]:
            hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFile, f'{region}', case)
            
            # note: case1 and 2 only apply to low bkg region
            # __________
            #|     | 3&4|
            #|     |    |
            #|      ----|
            #| Derive   |
            # ----------
            ###########################################
            if region!="V" and (case=="case1" or case=="case2"): # VR can be fully unblinded. unblind one year
                if case=="case1":
                    histFilePartner = ROOT.TFile.Open(f'hists_ABCDnn_case4_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
                    hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFilePartner, f'{region}', 'case4')
                else: # case2 partners with case3
                    histFilePartner = ROOT.TFile.Open(f'hists_ABCDnn_case3_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
                    hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFilePartner, f'{region}', 'case3')
            
                unblind_BpM_bin = hist_tgt_partner.GetXaxis().FindFixBin(unblind_BpM)
                unblind_ST_bin = hist_tgt_partner.GetYaxis().FindFixBin(unblind_ST)

                # set case1/2 upper right corner to case3/4
                for i in range(unblind_BpM_bin, Nbins_BpM_actual+1):
                    for j in range(unblind_ST_bin, Nbins_ST_actual+1):
                        hist_tgt.SetBinContent(i, j, hist_tgt_partner.GetBinContent(i,j))
                        hist_pre.SetBinContent(i, j, hist_pre_partner.GetBinContent(i,j))
                
                histFilePartner.Close()
            ##########################################

            
            #Then follow the standard way of creating correction hist
            # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
            hist_Correction = hist_tgt.Clone(f'BpMST_Correct{region}_{case}')
            hist_Correction.Add(hist_pre, -1.0)
            #hist_Correction.Smooth()
            hist_Correction.Divide(hist_pre)

            for i in range(Nbins_BpM_actual+1):
               for j in range(Nbins_ST_actual+1):
                   # set bin error to 0, so that the application correctly reflects the propogated change in bin error
                   hist_Correction.SetBinError(i,j,0)
                   # no correction on low stat bins 
                   if hist_tgt.GetBinContent(i,j)<=statCutoff: # TEMP
                   #    hist_Correction.SetBinContent(i,j,0) # TEMP
                       hist_Correction.SetBinContent(i,j,-1) # TEMP: reduce ABCDnn when mnr overpredicts

            histFile.cd()
            hist_Correction.Write(f'BpMST_Correct{region}')
            print(f'Saved BpMST_Correct{region} to {case}')

        histFile.Close()

    
# apply correction
def applyCorrection(corrType, region): # to D and V
    for case in ["case1", "case2", "case3", "case4"]:
        histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'UPDATE')

        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFile, region, case)

        # give clone names, so that ProfileX can distinguish them
        hist_preUp = hist_pre.Clone(f'preUp_{region}')
        hist_preDn = hist_pre.Clone(f'preDn_{region}')
        hist_cor = histFile.Get(f'BpMST_Correct{region}').Clone('cor')
        #hist_cor = histFile.Get(f'BpMST_Correct{corrType}').Clone('cor')x
        
        # pre = pre + pre*correction
        hist_cor.Multiply(hist_pre)
        hist_pre.Add(hist_cor)
        
        # preUp = pre + pre*2*correction
        hist_preUp.Add(hist_cor)
        hist_preUp.Add(hist_cor)
        
        # preDn = pre + pre*0, so do nothing

        # make sure that only lowST events are considered
        if region=="V" and (case=="case1" or case=="case2"):
            for i in range(Nbins_BpM_actual+1):
                for j in range(hist_preUp.GetYaxis().FindFixBin(validationCut),Nbins_ST_actual+1):
                    hist_pre.SetBinContent(i,j,0)
                    hist_preUp.SetBinContent(i,j,0)
                    hist_preDn.SetBinContent(i,j,0)
                    hist_pre.SetBinContent(i,j,0)
                    hist_preUp.SetBinError(i,j,0)
                    hist_preDn.SetBinError(i,j,0)
            
        hist_pre_1DUp = hist_preUp.ProjectionX()
        hist_pre_1DDn = hist_preDn.ProjectionX()
        hist_pre_1D = hist_pre.ProjectionX()
        
        hist_cor1D = hist_pre_1D.Clone()
        hist_cor1D.Add(hist_pre_1DDn, -1.0) # corrected - original
        hist_cor1D.Divide(hist_pre_1DDn)
    
        hist_pre_1D.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}')
        hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}Up')
        hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}Dn')
        hist_cor1D.Write(f'Bprime_mass_pre_Correct{corrType}on{region}')

        histFile.Close()

def applyTrainUncert(region):
    for case in ["case1", "case2", "case3", "case4"]:
        histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'UPDATE')
        
        hist_pre = getAlphaRatioTgtPreHists(histFile, region, case, False)
        hist_preUp = hist_pre.Clone(f'preUp_{region}')
        hist_preDn = hist_pre.Clone(f'preDn_{region}')
        
        hist_trainUncert = histFile.Get(f'BpMST_trainUncertfullST').Clone('trainUncert')

        # pre = pre +/- pre*trainUncert
        hist_trainUncert.Multiply(hist_preUp)
        hist_preUp.Add(hist_trainUncert, 1.0)
        hist_preDn.Add(hist_trainUncert, -1.0)

        if region=="V" and (case=="case1" or case=="case2"):
            for i in range(Nbins_BpM_actual+1):
                for j in range(hist_preUp.GetYaxis().FindFixBin(validationCut),Nbins_ST_actual+1):
                    hist_preUp.SetBinContent(i,j,0)
                    hist_preDn.SetBinContent(i,j,0)
                    hist_preUp.SetBinError(i,j,0)
                    hist_preDn.SetBinError(i,j,0)

        hist_pre_1DUp = hist_preUp.ProjectionX()
        hist_pre_1DDn = hist_preDn.ProjectionX()
        
        hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_trainUncertfullSTUp')
        hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_trainUncertfullSTDn')

        histFile.Close()

def applypNet(region):
    for case in ["case1", "case2"]:
        histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root','UPDATE')

        hist_pre      = histFile.Get(f'BpMST_pre_{region}').Clone('pre')
        hist_preUp = histFile.Get(f'Bprime_mass_pre_{region}_pNetUp').Clone('pNetUp') # TEMP: change to BpMST for newly run root files
        hist_preDn = histFile.Get(f'Bprime_mass_pre_{region}_pNetDn').Clone('pNetDn')

        # ST binning doesn't matter. Will get integrated
        hist_pre.RebinX(rebinX)
        hist_preUp.RebinX(rebinX)
        hist_preDn.RebinX(rebinX)

        # Upshift = (Upshifted - original)/original
        # Dnshift = (Dnshifted - original)/original
        hist_preUp.Add(hist_pre,-1.0)
        hist_preDn.Add(hist_pre,-1.0)

        hist_preUp.Divide(hist_pre)
        hist_preDn.Divide(hist_pre)

        # set bin error to 0, so that it acts like a pure scale factor
        for i in range(Nbins_BpM_actual+1):
            for j in range(Nbins_ST_actual+1):
                hist_preUp.SetBinError(i,j,0)
                hist_preDn.SetBinError(i,j,0)

        modifyOverflow2D(hist_pre)
        hist_pre.Scale(alphaFactors[case][region]["prediction"]/hist_pre.Integral())

        # shifted = shift*original + original (allow both shape and yield to change)
        hist_preUp.Multiply(hist_pre)
        hist_preDn.Multiply(hist_pre)

        hist_preUp.Add(hist_pre)
        hist_preDn.Add(hist_pre)
        
        if region=="V" and (case=="case1" or case=="case2"):
            for i in range(Nbins_BpM_actual+1):
                for j in range(hist_preUp.GetYaxis().FindFixBin(validationCut),Nbins_ST_actual+1):
                    hist_preUp.SetBinContent(i,j,0)
                    hist_preDn.SetBinContent(i,j,0)
                    hist_preUp.SetBinError(i,j,0)
                    hist_preDn.SetBinError(i,j,0)

        hist_pre_1DUp = hist_preUp.ProjectionX()
        hist_pre_1DDn = hist_preDn.ProjectionX()
        
        hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_pNetUp_1D')
        hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_pNetDn_1D')

        histFile.Close()

addHistograms()
for applyRegion in ['D', 'V']:
    applyCorrection('D', applyRegion) # D, V, V2
    applyTrainUncert(applyRegion)
    applypNet(applyRegion)
    
# plot histograms
for case in ["case1", "case2", "case3", "case4"]:
    plotHists2D_Separate(case)

#plotHists2D_All() # this function needs some work. Complains about merging hists with diff bins
