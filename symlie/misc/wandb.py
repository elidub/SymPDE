import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import wandb


def get_inspectdev_df(tags: list):
    api = wandb.Api()
    runs = api.runs('eliasdubbeldam/symlie')

    runs_inspect = []
    for run in runs:
        if (run.state != 'finished'): continue
        if not all(tag in run.tags for tag in tags): continue
        runs_inspect.append(run)

    inspect_df = new_runs(runs_inspect).reset_index(drop=True)

    return inspect_df


def get_inspect_df(reload: bool = False, results_file: str = '../logs/store/inspect_df.pkl'):

    if not reload:
        return pd.read_pickle(results_file)

    api = wandb.Api()
    runs = api.runs('eliasdubbeldam/symlie')

    runs_inspect = []
    for run in runs:
        if (run.state != 'finished'): continue
        if not ('inspect' in run.tags): continue
        runs_inspect.append(run)

    inspect_df = new_runs(runs_inspect).reset_index(drop=True)

    inspect_df.to_pickle(os.path.join(results_file))

    return inspect_df

def exceptions(runs, results_df_old = None):
    
    runs_new = []
    for run in runs:

        if (run.state != 'finished'): continue
        if results_df_old is not None:
            if (run.id in results_df_old['run_id'].values): continue 
        if 'dev' in run.tags: continue
        # if not ('new' in run.tags or 'hparam' in run.tags): continue
        if not ('new' in run.tags): continue
        
        runs_new.append(run)
    return runs_new

def new_runs(runs):
    config_list = []

    pbar = tqdm(runs)
    for run in pbar:
        id = run.id
        pbar.set_description(f'Retreiving wandb {str(id)}')

        # Retreive config from wandb run, add all info to config
        config = run.config

        # Add run id and name to config
        config['run_id']   = id
        config['run_name'] = run.name
        config['tags']     = run.tags

        # Test loss
        losses = ['test_loss_o', 'test_loss_dg', 'test_loss_dx', 'test_loss_do', 'test_loss_do_a', 'test_loss_do_b', 'test_loss_do_a_mmd', 'test_loss_do_b_mmd']
        # losses = ['test_loss']
        for loss in losses:
            try:
                test_loss_history = run.history(keys=[loss])[loss]
                if len(test_loss_history) != 1:
                    print(f'Warning: {id} has test_loss {test_loss_history}')
                config[loss] = test_loss_history.item()
            except:
                config[loss] = np.nan
                print(f'Warning: {loss} not found')

        config_list.append(config)

    results_df = pd.DataFrame(config_list)
    return results_df

def update_results_df(from_scratch: bool = False, results_file: str = '../logs/store/results_df.pkl'):
    api = wandb.Api()
    runs = api.runs('eliasdubbeldam/symlie')

    if from_scratch:
        runs_selected = exceptions(runs)
    
        print(len(runs_selected))
        results_df_new = results_df = new_runs(runs_selected).reset_index(drop=True)
    else:
        results_df_old = pd.read_pickle(results_file)
        runs_selected = exceptions(runs, results_df_old)

        print(len(runs_selected))
        results_df = new_runs(runs_selected)
        results_df_new = pd.concat([results_df_old, results_df]).reset_index(drop=True)
    
    # assert_unique(results_df_new)
    results_df_new.to_pickle(os.path.join(results_file))