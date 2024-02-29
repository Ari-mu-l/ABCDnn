# updated 10/25 by Daniel Li
# python train_abcdnn_hpo.py -s rootFiles_BprimeSTCase14/OctMajor_all_mc_p100.root -b rootFiles_BprimeST/Case14/OctMinor_all_mc_p100.root -t rootFiles_BprimeST/Case14/OctData_all_data_p100.root -f 0.6 -m random3

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from json import loads as load_json
from json import dumps as write_json
from argparse import ArgumentParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import custom methods
import config
import abcdnn

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-b", "--minor",  required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "-f", "--fraction", required = True )
parser.add_argument( "--hpo", action = "store_true" )
parser.add_argument( "--randomize", action = "store_true" )
parser.add_argument( "--verbose", action = "store_true" )
parser.add_argument( "--closure", action = "store_true" )
parser.add_argument( "-m", "--modeltag", default = "best_model", help = "Name of model saved to Results directory" )
parser.add_argument( "-d", "--disc_tag", default = "ABCDnn", help = "Postfix appended to original branch names of transformed variables" )
args = parser.parse_args()

if args.randomize: config.params["MODEL"]["SEED"] = np.random.randint( 100000 )

with open('{}/hp_{}.txt'.format('Results', args.modeltag), 'r') as hp_file:
  content = hp_file.read().split(', ')
  print(content)
  #print(content[0].split(': ')[-1])
exit()

hp["NODES_COND"] = content[0].split(': ')[-1]
hp["HIDDEN_COND"] = content[1].split(': ')[-1]
hp["NODES_TRANS"] = content[2].split(': ')[-1]
hp["LRATE"] = content[3].split(': ')[-1]
hp["DECAY"] = content[4].split(': ')[-1]
hp["GAP"] = content[5].split(': ')[-1]
hp["DEPTH"] = content[6].split(': ')[-1]
hp["REGULARIZER"] =
hp["BETA1"] = content[7].split(': ')[-1]
hp["BETA2"] = content[8].split(': ')[-1]
hp_MMD_SIGMAS = (content[9].split(': ')[-1]).split(', ')
hp["MMD SIGMAS"] = [hp_MMD_SIGMAS[-3], hp_MMD_SIGMAS[-2],hp_MMD_SIGMAS[-1]]
hp["MMD WEIGHTS"] = content[10].split(': ')[-1]
hp["MINIBATCH"] = 2048
hp["LRATE"] = float(args.modeltag.split('_')[-3])
hp["DECAY"] = float(args.modeltag.split('_')[-2])
hp["GAP"] = int(args.modeltag.split('_')[-1])
hp["REGULARIZER"] = args.modeltag.split('_')[-2]
hp["ACTIVATION"] = args.modeltag.split('_')[-1]
#print("hp: {}".format(hp))
#with open('{}/hp_{}.txt'.format('Results', args.modeltag), 'w') as hp_file:
#  hp_file.write(str(hp))    
#print(type(hp["MINIBATCH"]))
#for param in hp:
#	print("{}: {}, type: {}".format(param, hp[param], type(hp[param])))


print( "[START] Training ABCDnn model {} iwth discriminator: {}".format( args.modeltag, args.disc_tag ) )
if config.params[ "MODEL" ][ "RETRAIN" ]:
  print( "[INFO] Deleting existing model with name: {}".format( args.modeltag ) )
  os.system( "rm -rvf Results/{}*".format( args.modeltag ) )
if args.hpo:
  print( "  [INFO] Running on optimized parameters" )
  with open( os.path.join( config.results_path, "opt_params.json" ), "r" ) as jsf:
    hpo_cfg = load_json( jsf.read() )
    for key in hpo_cfg["PARAMS"]:
      hp[key] = hpo_cfg["PARAMS"][key]
else:
  print( "  [INFO] Running on fixed parameters" )
for key in hp: print( "   - {}: {}".format( key, hp[key] ) )
               
abcdnn_ = abcdnn.ABCDnn_training()
abcdnn_.setup_events(
  rSource = args.source, 
  rMinor  = args.minor,
  rTarget = args.target,
  selection = config.selection,
  variables = config.variables,
  regions = config.regions,
  closure = args.closure,
  frac = float(args.fraction)
)

abcdnn_.setup_model(
  nodes_cond = hp[ "NODES_COND" ],
  hidden_cond = hp[ "HIDDEN_COND" ],
  nodes_trans = hp[ "NODES_TRANS" ],
  lr = hp[ "LRATE" ],
  decay = hp[ "DECAY" ],
  gap = hp[ "GAP" ],
  depth = hp[ "DEPTH" ],
  regularizer = hp[ "REGULARIZER" ],
  initializer = hp[ "INITIALIZER" ],
  activation = hp[ "ACTIVATION" ],
  beta1 = hp[ "BETA1" ],
  beta2 = hp[ "BETA2" ],
  mmd_sigmas = hp[ "MMD SIGMAS" ],
  mmd_weights = hp[ "MMD WEIGHTS" ],
  minibatch = hp[ "MINIBATCH" ],
  savedir = config.params[ "MODEL" ][ "SAVEDIR" ],
  disc_tag = args.disc_tag,
  seed = config.params[ "MODEL" ][ "SEED" ],
  verbose = config.params[ "MODEL" ][ "VERBOSE" ],
  model_tag = args.modeltag,
  closure = config.params[ "MODEL" ][ "CLOSURE" ],
  permute = config.params[ "MODEL" ][ "PERMUTE" ],
  retrain = config.params[ "MODEL" ][ "RETRAIN" ]
)

abcdnn_.train(
  steps = config.params[ "TRAIN" ][ "EPOCHS" ],
  patience = config.params[ "TRAIN" ][ "PATIENCE" ],
  monitor = config.params[ "TRAIN" ][ "MONITOR" ],
  display_loss = config.params[ "TRAIN" ][ "SHOWLOSS" ],
  early_stopping = config.params[ "TRAIN" ][ "EARLY STOP" ],
  monitor_threshold = config.params[ "TRAIN" ][ "MONITOR THRESHOLD" ],
  periodic_save = config.params[ "TRAIN" ][ "PERIODIC SAVE" ],
)

abcdnn_.evaluate_regions()
abcdnn_.extended_ABCD()
abcdnn_.save_hyperparameters()
