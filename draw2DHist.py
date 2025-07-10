import ROOT
import math

year = 'all'
#sample = ['Major', 'Signal']
rfileMajor = ROOT.TFile.Open(f'rootFiles_logBpMlogST_Jan2025Run2/Case14/JanMajor_{year}_mc_p100.root','READ')
rfileSignal = ROOT.TFile.Open(f'rootFiles_logBpMlogST_Jan2025Run2/Case14/JanSignal_{year}_mc_p100.root','READ')

#bkgList = ['ewk','qcd','ttbar','ttx','wjets','singletop']

hist2DMajor = ROOT.TH2D('N_b_vs_N_forward_Major','N_b_vs_N_forward_Major',8,0,8,8,0,8)
hist2DSignal = ROOT.TH2D('N_b_vs_N_forward_Signal','N_b_vs_N_forward_Signal',8,0,8,8,0,8)
hist2DSignificance = ROOT.TH2D('N_b_vs_N_forward_Significance','N_b_vs_N_forward_Significance',8,0,8,8,0,8)

tTreeMajor = rfileMajor.Get('Events')
tTreeSignal = rfileSignal.Get('Events')
for evt in tTreeMajor:
    hist2DMajor.Fill(evt.NJets_forward,evt.NJets_DeepFlavL)
print('Major histogram created.')

for evt in tTreeSignal:
    hist2DSignal.Fill(evt.NJets_forward,evt.NJets_DeepFlavL)
print('Signal histogram created.')

c1 = ROOT.TCanvas('c1','c1')
hist2DMajor.Draw('COLZ')
c1.SaveAs(f'N_b_vs_N_forward_Major.png')

c2 = ROOT.TCanvas('c2','c2')
hist2DSignal.Draw('COLZ')
c2.SaveAs(f'N_b_vs_N_forward_Signal.png')

for i in range(1,9):
    for j in range(1,9):
        if hist2DMajor.GetBinContent(i,j)!=0:
            hist2DSignificance.SetBinContent(i,j,hist2DSignal.GetBinContent(i,j)/math.sqrt(hist2DMajor.GetBinContent(i,j)))
        else:
            hist2DSignificance.SetBinContent(i,j,0)

c3 = ROOT.TCanvas('c3','c3')
hist2DSignificance.Draw('COLZ')
c3.SaveAs(f'N_b_vs_N_forward_Significance.png')

rfileMajor.Close()
rfileSignal.Close()


