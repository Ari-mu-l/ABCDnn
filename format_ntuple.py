# this script takes an existing step3 ROOT file and formats it for use in the ABCDnn training script
# formats three types of samples: data (no weights), major MC background (no weights), and minor MC backgrounds (weights)
# last modified April 11, 2023 by Daniel Li

import os, sys, ROOT
from array import array
from argparse import ArgumentParser
import config
import tqdm
import xsec
from utils import *
from samples import *

parser = ArgumentParser()
parser.add_argument( "-y",  "--year", default = "2018UL", help = "Year for sample" )
parser.add_argument( "-n", "--name", required = True, help = "Output name of ROOT file" )
parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "Bprime_mass", "gcJet_ST" ], help = "Variables to transform" )
parser.add_argument( "-p",  "--pEvents", default = 100, help = "Percent of events (0 to 100) to include from each file." )
parser.add_argument( "-l",  "--location", default = "LPC", help = "Location of input ROOT files: LPC,BRUX" )
parser.add_argument( "--doMajorMC", action = "store_true", help = "Major MC background to be weighted using ABCDnn" )
parser.add_argument( "--doMinorMC", action = "store_true", help = "Minor MC background to be weighted using traditional SF" )
parser.add_argument( "--doClosureMC", action = "store_true", help = "Closure MC background weighted using traditional SF" )
parser.add_argument( "--doData", action = "store_true" )
parser.add_argument( "--JECup", action = "store_true", help = "Create an MC dataset using the JECup shift for ttbar" )
parser.add_argument( "--JECdown", action = "store_true", help = "Create an MC dataset using the JECdown shift for ttbar" )
args = parser.parse_args()

if args.location not in [ "LPC", "BRUX" ]: quit( "[ERR] Invalid -l (--location) argument used. Quitting..." )

# FIXME
# might need to move backwards
#print( "[INFO] Evaluating cross section weights." )
#weightXSec = {}
#if args.doMinorMC:
#  files = config.samples_input[ args.year ][ "MINOR MC" ] 
#elif args.doMajorMC: 
#  files = config.samples_input[ args.year ][ "MAJOR MC" ]
#elif args.doClosureMC:
#  files = config.samples_input[ args.year ][ "CLOSURE" ] 
#elif args.doData:
#  files = config.samples_input[ args.year ][ "DATA" ]
#else:
#  quit( "[ERR] No valid option used, choose from doMajorMC, doMinorMC, doClosureMC, doData" )
#for f in files:
#  if args.doMinorMC or args.doMajorMC or args.doClosureMC:
#    weightXSec[f] = xsec.lumi[ args.year ] * xsec.xsec[f] / xsec.nRun[f] # FIXME # Check what nRun corresponds to
#    print( ">> {}: {:.1f} x {:.5f} = {:.1f}".format( f.split("_")[0], nRun[f], weightXSec[f], xsec.lumi[ args.year ] * xsec.xsec[f] ) )
#    rF_.Close()
#  else:
#    weightXSec[f] = 1.


#ROOT.gInterpreter.Declare("""
#float compute_weight_minor( float scale, float elRecoSF, float leptonIDSF, float leptonHLTSF, float xsecEff) {
#return scale * elRecoSF * leptonIDSF * leptonHLTSF * xsecEff;
#}
#""") 

ROOT.gInterpreter.Declare("""
    float compute_weight( float genWeight, float lumi, float xsec, float nRun ){
    return genWeight * lumi * xsec / (nRun * abs(genWeight));
    }
    """) #FIXME add SFs later

#ROOT.gInterpreter.Declare("""  
#    float compute_weight( float scale, float lumi, float xsec, float nRun, float genWeight, float elRecoSF, float leptonIDSF, float leptonHLTSF ) {
#    return scale * elRecoSF * leptonIDSF * leptonHLTSF * genWeight * lumi * xsec / (nRun * abs(genWeight));
#}
#""") 
#FIXME add more SFs

class ToyTree:
  def __init__( self, name, trans_var ):
    # trans_var is transforming variables
    self.name = name
    self.rFile = ROOT.TFile.Open( "{}.root".format( name ), "RECREATE" )
    self.rTree = ROOT.TTree( "Events", name )
    self.variables = { # variables that are used regardless of the transformation variables
      "xsecWeight": { "ARRAY": array( "f", [0] ), "STRING": "xsecWeight/F" } # might not needed. # CHECK
    }
    for variable in config.variables.keys():
      if not config.variables[ variable ][ "TRANSFORM" ]:
        self.variables[ variable ] = { "ARRAY": array( "i", [0] ), "STRING": str(variable) + "/I" } # MODIFY DATATYPE
    
    for variable in trans_var:
      self.variables[ variable ] = { "ARRAY": array( "f", [0] ), "STRING": "{}/F".format( variable ) }
   
    self.selection = config.selection 
    
    for variable in self.variables:
      self.rTree.Branch( str( variable ), self.variables[ variable ][ "ARRAY" ], self.variables[ variable ][ "STRING" ] ) # create a tree with only useful branches
    
  def Fill( self, event_data ): # fill all tree branches for each event
    for variable in self.variables:
      self.variables[ variable ][ "ARRAY" ][0] = event_data[ variable ]
    self.rTree.Fill()
  
  def Write( self ):
      print( ">> Writing {} entries to {}.root".format( self.rTree.GetEntries(), self.name ) )
      self.rFile.Write()
      self.rFile.Close()
      
def format_ntuple( inputs, output, trans_var):
  sampleDir = config.sampleDir[ args.year ]
  if ( args.JECup or args.JECdown ) and "data" in output:
    print( "[WARNING] Ignoring JECup and/or JECdown arguments for data" )
  elif args.JECup and not args.JECdown:
    print( "[INFO] Running with JECup samples" )
    sampleDir = sampleDir.replace( "nominal", "JECup" )
    output = output.replace( "mc", "mc_JECup" )
  elif args.JECdown and not args.JECup:
    print( "[INFO] Running with JECdown samples" )
    sampleDir = sampleDir.replace( "nominal", "JECdown" )
    output = output.replace( "mc", "mc_JECdown" )
  elif args.JECdown and args.JECup:
    sys.exit( "[WARNING] Cannot run with both JECup and JECdown options. Select only one or none. Quitting..." )
  
  ntuple = ToyTree( output, trans_var )

  for sample in inputs:
    samplename = sample.samplename.split('/')[1]
    print( ">> Processing {}".format( samplename ) )
    fChain = readTreeNominal(samplename,config.sourceDir["LPC"],"Events") # read rdf for processing
    rDF = ROOT.RDataFrame(fChain)
    sample_total = rDF.Count().GetValue()
    filter_string = "" 
    scale = 1. / ( int( args.pEvents ) / 100. ) # isTraining == 3 is 20% of the total dataset # COMMENT: What is isTraining? # what is scale used for?
    for variable in ntuple.selection: 
      for i in range( len( ntuple.selection[ variable ]["CONDITION"] ) ):
        if filter_string == "": 
          filter_string += "( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
        else:
          filter_string += "|| ( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
    #print("filter_string: {}".format(filter_string))
    #if args.year == "2018" and args.doData:
    #  filter_string += " && ( leptonEta_MultiLepCalc > -1.3 || ( leptonPhi_MultiLepCalc < -1.57 || leptonPhi_MultiLepCalc > -0.87 ) )" 
    rDF_filter = rDF.Filter( filter_string )
    rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( {}, {}, {}, {} )".format(scale, xsec.lumi[args.year], sample.xsec, sample.nrun) )
    sample_pass = rDF_filter.Count().GetValue() # number of events passed the selection
    dict_filter = rDF_weight.AsNumpy( columns = list( ntuple.variables.keys() + [ "xsecWeight" ] ) ) # useful rdf branches to numpy
    del rDF, rDF_filter, rDF_weight
    n_inc = int( sample_pass * float( args.pEvents ) / 100. ) # get a specified portion of the passed events
 
    for n in tqdm.tqdm( range( n_inc ) ):
      event_data = {}
      for variable in dict_filter:
        event_data[ variable ] = dict_filter[ variable ][n] 

      ntuple.Fill( event_data )

    print( ">> {}/{} events saved...".format( n_inc, sample_total ) )
  ntuple.Write()
  
if args.doMajorMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "MAJOR MC" ], output = args.name + "_" + args.year + "_mc_p" + args.pEvents, trans_var = args.variables )
elif args.doMinorMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "MINOR MC" ], output = args.name + "_" + args.year + "_mc_p" + args.pEvents, trans_var = args.variables )
elif args.doClosureMC:
  format_ntuple( inputs = config.samples_input[ args.year ][ "CLOSURE" ], output = args.name + "_" + args.year + "_mc_p" + args.pEvents, trans_var = args.variables )
if args.doData:
  format_ntuple( inputs = config.samples_input[ args.year ][ "DATA" ], output = args.name + "_" + args.year + "_data_p" + args.pEvents, trans_var = args.variables )
