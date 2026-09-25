"""Compile a completed seed's raw Train bank into an independent Frozen candidate."""
import argparse
import json
from atomic_skillgraph.deployment.train_bank_compiler import compile_train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(compile_train(args.config,args.output),indent=2))
