## Installation

```bash
conda create -n sympde python=3.10
conda activate sympde
pip install -r requirements.txt
# conda install -c conda-forge fortran-compiler # for escnn
```

## Repository

### External repositories
The folder `ext_repos/` contains cloned repositories from other authors. I removed the `.git` file in those repo's so I can add them to this repo `SymPDE`.

## Presentations
- [Presentation 13 dec](assets/presentations/presentation_13dec.pdf)
- [Presentation 23 nov](assets/presentations/presentation_23nov.html) (still in html)

## Commands
Local
```bash
ss  sh ~/EliasMBA/Projects/Uni/SymPDE/jobs/sync.sh
ca  conda activate sympde
ru  python run.py
ruc python run.py --config $1
```

Snellius
```bash
ll  ls -la
sq  squeue -u eliasd
squ squeue
sb  sbatch jobs/sumlie_train_array.job
sba sh jobs/run_all.sh
sbals sh jobs/job_arrays
ru  python run.py
ruc python run.py --config $1
rul python run.py --logger None $1
ca  conda activate sympde
catb    conda activate sympde ; tensorboard --logdir logs
cl      gpu debugger 10 minutes
cl1h    gpu debugger 1 hour
cl8h    gpu debugger 8 hours
```
