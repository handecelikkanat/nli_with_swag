# nli_with_swag
Using Stochastic Weight Averaging-Gaussian method for uncertainty representation (https://github.com/wjmaddox/swa_gaussian/blob/master/README.md), for Natural Language Inference with Transformer-based architectures

# Installation
1. Install dependencies as specified in requirements.txt:
```
pip install -r requirements.txt
```

2. Install [swa_gaussian](https://github.com/wjmaddox/swa_gaussian/)
```
pip install git+https://github.com/wjmaddox/swa_gaussian.git@ed5fd56e34083b42630239e59076952dee44daf4
```

3. Clone [nli-with-transformers](https://github.com/aarnetalman/nli-with-transformers)
```
git clone https://github.com/aarnetalman/nli-with-transformers src/nli_with_transformers
```

# File Structure

```
.
+-- src/
|   +-- nli_with_transformers
|   +-- run_nli_with_swag.py (main runner code)
|   +-- utils.py (overriden utility functions from swa_gaussian)
|   +-- models.py (language models wrapper in swa_gaussian style)

+-- experiments/
|   +-- logs (folder for sbatch output and error file redirections)
|   +-- run_sbatch.sh (sbatch submission script)
|   +-- run_interactive.sh (bash script for running in interactive mode)
|   +-- grad_cov/ (gradient covariance and optimal learning rate experiments)      

+-- requirements.txt
+-- README.md
+-- LICENSE
```

# Running Experiments
Experiments can be run in one of the following ways:

1. Using batch submission script:
```
sbatch experiments/run_sbatch.sh
```

2. Using interactive run script:
``` 
srun -n1 -tDURATION --partition=PARTITION --gres=RESOURCE --mem=MEM --account=<PROJECT_NAME>
bash experiments/run_interactive.sh
```

3. Invoking the runner code directly, eg.:
``` 
python $SRCDIR/run_nli_with_swag.py \
    --swa \
    --model Bert \
    --dir /scratch/project_2002233/SWAG/nli_with_swag/outputs/1 \
    --lr_init 0.001 \
    --batch_size 128 \
    --seed 199
```

Important parameters are:

```
run_nli_with_sway.py:
+-- args.max_num_models (Maximum number of models to collect (older models are dropped after this number is exceeded), 
                         equals to the rank of SWAG covariance matrix approximation)
+-- args.swa_c_epochs (Frequency in epochs of SWA model collection)
```

# References
[swa_gaussian](https://github.com/wjmaddox/swa_gaussian/)

[nli-with-transformers](https://github.com/aarnetalman/nli-with-transformers)
