import os, sys
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), '../symlie'))
from misc.utils_arrays import write_lines, clean_val, dict_to_array

def main(job_dir, array_file, array_dir):
    df = pd.read_csv(array_file, dtype=object)
    df = df.set_index('experiment')
    df.head()

    df = pd.DataFrame({key : vals.apply(lambda val: clean_val(val)) for key, vals in df.items() if key != 'experiment'})

    # select only data_kwargs and transform_kwargs
    # df = df[['y_high', 'y_low', 'noise_std', 'grid_size', 'eps_mult', 'data_dir']]

    for experiment, hparams in df.iterrows():
        output_file = os.path.join(array_dir, experiment + '.txt')
        output_lines = dict_to_array(hparams.dropna().to_dict())
        
        n_runs = output_lines.count('\n') + 1
        print(f"Writing {experiment} with {n_runs} lines")
        
        write_lines(output_file, output_lines)




if __name__ == '__main__':
    job_dir = '../jobs'
    array_file = os.path.join(job_dir, 'arrays.csv')
    array_dir  = os.path.join(job_dir, 'arrays-v2')
    main(job_dir, array_file, array_dir)
