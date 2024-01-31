from subprocess import call
import os
import argparse


def parse_options(notebook=False):
    parser = argparse.ArgumentParser(description='SymLieTrainArray')

    parser.add_argument("array", type = str)
    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def main(args):

    print(os.listdir('.'))


    # read txt file sine1d.txt
    with open(os.path.join('jobs', 'arrays', args.array + '.txt'), 'r') as f:
        run_args = f.readlines()

    print(f"Running {len(run_args)} jobs from {args.array}!")

    os.chdir('symlie')
    for run_arg in run_args:
        # print(run_args)
        run_string = f"python run.py {run_arg}"
        print(run_string)
        call(run_string, shell=True)

        

    # print(os.listdir('.'))
    # call(["python", "jobs/symlie_train_array.py", "--array", args.array])


if __name__ == "__main__":
    args = parse_options()
    main(args)
