# TODO: check validation prediction and uncertainty
# Did it quick for inaugural talk

import os, sys
import numpy as np
import ROOT
import json

if len(sys.argv)>1:
    getAlphaRatio = sys.argv[1]
else:
    getAlphaRatio = "True"

if not os.path.exists('validation_plots'):
    os.makedirs('validation_plots')

###############################
# calculate correction factor #
###############################
if getAlphaRatio=="True":
    # get counts
    caseName = {"case1" : "tagTjet",
                "case2" : "tagWjet",
                "case3" : "untagTlep",
                "case4" : "untagWlep",
                }

    counts = {"case14":{},
              "case23":{},
              "case1":{},
              "case2":{},
              "case3":{},
              "case4":{},
              }

    for case in counts:
        #for region in ["A", "B", "C", "D", "X", "Y", "V", "BV"]:
        for region in ["B", "D", "V", "BV"]:
            counts[case][region] = {}

    def getCounts(case, region):
        print(f'Processing {case}')
        tempFileName  = f'/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates/templates{region}_Oct2024_42bins/templates_BpMass_138fbfb.root'
        tFile = ROOT.TFile.Open(tempFileName, 'READ')
        hist_data  = tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__data_obs')
        hist_major = tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__qcd') + tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__wjets') + tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__singletop') + tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__ttbar')
        hist_minor = tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__ttx') + tFile.Get(f'BpMass_138fbfb_isL_{caseName[case]}_{region}__ewk')
        
        # integrate from the bin where 400 is in to the last bin
        #counts[case][region]["data"]  = hist_data.Integral(hist_data.FindBin(400), 99)
        #counts[case][region]["major"] = hist_major.Integral(hist_major.FindBin(400), 99)
        #counts[case][region]["minor"] = hist_minor.Integral(hist_minor.FindBin(400), 99)

        counts[case][region]["data"]  = hist_data.Integral()
        counts[case][region]["major"] = hist_major.Integral()
        counts[case][region]["minor"] = hist_minor.Integral()
        
        if region=="D" or region=="V":
            counts[case][region]["unweighted"] = hist_major.GetEntries()

        tFile.Close()

    #for region in ["A", "B", "C", "D", "X", "Y", "V", "BV"]:
    for region in ["B", "D", "V", "BV"]:
        print(f'Getting counts for region {region}')
        getCounts("case1", region)
        getCounts("case4", region)
        getCounts("case2", region)
        getCounts("case3", region)

        counts["case14"][region]["data"]  = counts["case1"][region]["data"]  + counts["case4"][region]["data"]
        counts["case23"][region]["data"]  = counts["case2"][region]["data"]  + counts["case3"][region]["data"]

        counts["case14"][region]["major"] = counts["case1"][region]["major"] + counts["case4"][region]["major"]
        counts["case23"][region]["major"] = counts["case2"][region]["major"] + counts["case3"][region]["major"]

        counts["case14"][region]["minor"] = counts["case1"][region]["minor"] + counts["case4"][region]["minor"]
        counts["case23"][region]["minor"] = counts["case2"][region]["minor"] + counts["case3"][region]["minor"]

        if region=="D" or region=="V":
            counts["case14"][region]["unweighted"]  = counts["case1"][region]["unweighted"]  + counts["case4"][region]["unweighted"]
            counts["case23"][region]["unweighted"]  = counts["case2"][region]["unweighted"]  + counts["case3"][region]["unweighted"]

    # store counts in a json file
    print("Writing to counts.json...")
    json_object = json.dumps(counts, indent=4)
    with open("counts.json", "w") as outfile:
        outfile.write(json_object)

        
    # alpha-ratio prediction
    yield_pred = {"case14":{"D":{}, "V":{}},
                  "case23":{"D":{}, "V":{}},
                  "case1":{"D":{}, "V":{}},
                  "case2":{"D":{}, "V":{}},
                  "case3":{"D":{}, "V":{}},
                  "case4":{"D":{}, "V":{}},
                  }

    def getPrediction(case, region):
        if region=="V":
            B = "BV"
            D = "V"
        else:
            B = "B"
            D = "D"
        print(f'Getting prediction for {case}...')
        target_B    = counts[case][B]["data"] - counts[case][B]["minor"]
        predict_D   = target_B * counts[case][D]["major"] / counts[case][B]["major"]
        predict_val = target_B * counts[case][region]["major"] / counts[case][B]["major"]
        target_val  = counts[case][region]["data"] - counts[case][region]["minor"]

        yield_pred[case][region]["prediction"]  = predict_D
        yield_pred[case][region]["factor"]      = predict_D / counts[case][D]["unweighted"]
        yield_pred[case][region]["systematic"]  = predict_D * np.sqrt(1/target_B + 1/counts[case][B]["major"] + 1/counts[case][D]["major"])
        yield_pred[case][region]["statistical"] = np.sqrt(predict_D)
        yield_pred[case][region]["closure"]     = abs(predict_val-target_val)
        yield_pred[case][region]["uncertainty"] = np.sqrt(yield_pred[case][region]["systematic"]**2 + yield_pred[case][region]["statistical"]**2 + yield_pred[case][region]["closure"]**2) / predict_D

        print('Data:{}'.format(counts[case]["D"]["data"]))
        print('Minor:{}'.format(counts[case]["D"]["minor"]))
        print(f'Data-Minor:{target_val}')

        print('Major from MC         :{}'.format(counts[case]["D"]["major"]))
        print(f'Major from alpha-ratio:{predict_D}')

        print('Major deviation in MC: {}%'.format(round(100*(abs(counts[case][region]["major"]-target_val)/target_val),2)))
        print(f'Major deviation in alpha-ratio: {round(100*(yield_pred[case][region]["closure"]/target_val),2)}%')
        print(f'Total uncertainty from alpha-ratio (percentage): {round(100*yield_pred[case][region]["uncertainty"],2)}%\n')


    print("\nPerforming alpha-ratio estimation.\n")
    for case in ["case14", "case23", "case1", "case2", "case3", "case4"]:
        for region in ["V", "D"]:
            getPrediction(case, region)

    # write alpha-ratio restuls to a json file
    print("Writing to alphaRatio_factors.json...")
    json_object = json.dumps(yield_pred, indent=4)
    with open("alphaRatio_factors.json", "w") as outjson:
        outjson.write(json_object)
else:
    print("Skipping counts and alpha-ratio estimation...")


exit()

# Load histograms and plot validation
binlo = 400
binhi = 2500
bins = 42

def plot_validation(case, shape):
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
    
    hist_data    = histFile.Get(f'Bprime_mass_data_val')
    hist_minor   = histFile.Get(f'Bprime_mass_minor_val')
    hist_ABCDnn  = histFile.Get(f'Bprime_mass_ABCDnn_val')
    
    c = ROOT.TCanvas("")
    legend = ROOT.TLegend(0.6,0.6,0.8,0.7)
    legend.SetBorderSize(0)
    hist_ABCDnn.SetLineColor(ROOT.kRed)
    
    # test shape
    if shape:
        hist_dataNominor = hist_data - hist_minor
        hist_dataNominor.Scale(1/hist_dataNominor.Integral())
        hist_ABCDnn.Scale(1/hist_ABCDnn.Integral())
        
        legend.AddEntry(hist_dataNominor, "data-minor", "l")
        legend.AddEntry(hist_ABCDnn, "ABCDnn", "l")
    
        hist_dataNominor.Draw("HIST")
        hist_ABCDnn.Draw("HIST SAME")
        legend.Draw()
        c.SaveAs(f'validation_plots/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_shape.png')
    else:
        #hist_predict = hist_minor + hist_ABCDnn
    
        #hist_ABCDnn.Scale(yield_pred[case]/hist_ABCDnn.Integral())
        
        #ratio = hist_predict/hist_data
        #ratio.Print("all")
        print(hist_ABCDnn.Integral())
        print(hist_dataNominor.Integral())
        print((hist_ABCDnn.Integral()-hist_dataNominor.Integral())/hist_dataNominor.Integral())

        #legend.AddEntry(hist_data, "data", "l")
        #legend.AddEntry(hist_predict, "ABCDnn+minor", "l")
        legend.AddEntry(hist_ABCDnn, "ABCDnn", "l")
        legend.AddEntry(hist_dataNominor, "data-minor", "l")

        #hist_predict.SetLineColor(ROOT.kRed)
        #hist_predict.Draw("HIST")
        hist_ABCDnn.Draw("HIST")
        hist_dataNominor.Draw("HIST SAME")
        #hist_data.Draw("HIST SAME")
        
        legend.Draw()
        c.SaveAs(f'validation_plots/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.png')


#plot_validation("case14")
plot_validation("case23", False)
