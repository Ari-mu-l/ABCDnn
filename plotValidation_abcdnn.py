import os
import numpy as np
from ROOT import *

if not os.path.exists('validation_plots'):
    os.makedirs('validation_plots')

case = "case23"
binlo = 400
binhi = 2500
bins = 43
histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")

hist_data    = histFile.Get(f'Bprime_mass_data_val')
hist_minor   = histFile.Get(f'Bprime_mass_minor_val')
hist_ABCDnn  = histFile.Get(f'Bprime_mass_ABCDnn_val')
hist_predict = hist_minor + hist_ABCDnn

c = TCanvas("")
hist_predict.SetFillColor(-7)

hist_data.Draw("E")
hist_predict.Draw("SAME")

c.SaveAs('validation_plots/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.png')


