from tqdm import tqdm
import pandas as pd
import numpy as np

def exceptions(runs, results_df_old):
    
    runs_new = []
    for run in runs:

        if (run.state != 'finished'): continue
        if (run.id in results_df_old['run_id'].values): continue 
        if 'dev' in run.tags: continue
        
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
        try:
            test_loss_history = run.history(keys=['test_loss'])['test_loss']
            if len(test_loss_history) != 1:
                print(f'Warning: {id} has test_loss {test_loss_history}')
            config['test_loss'] = test_loss_history.item()
        except:
            config['test_loss'] = np.nan

        config_list.append(config)

    results_df = pd.DataFrame(config_list)
    return results_df