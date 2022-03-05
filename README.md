![ABAS](https://github.com/lr94/abas/blob/master/abas.png?newteaser)

# ABAS

Code release for **[Adversarial Branch Architecture Search for Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/document/9706889)**.

If you use this code or the attached files for research purposes, please cite
```
@inproceedings{robbiano2021adversarial,
	title        = {Adversarial Branch Architecture Search for Unsupervised Domain Adaptation},
	author       = {Robbiano, Luca and Ur Rahman, Muhammad Rameez and Galasso, Fabio and Caputo, Barbara and Carlucci, Fabio Maria},
	year         = 2022,
	booktitle    = {2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
	volume       = {},
	number       = {},
	pages        = {1008--1018},
	doi          = {10.1109/WACV51458.2022.00108}
}
```

### Software requirements
* CUDA
* Python 3.6 or newer
* PyTorch 1.6 or newer
* Other Python libraries listed in `requirements.txt`

### Hardware requirements
* 10 GB available on each GPU
* Optional but strongly recommended: a cluster capable of running at least 8 parallel GPU jobs

### Run experiments

To launch an ABAS run (OfficeHome, source Art, target Clipart):

```
./scripts/launch_slurm_stub.sh \
  --source art-oh \
  --target clipart-oh \
  --criterion 'regression(regressors/regr_no-pseudolabels_for_oh.pkl)' \
  --run-criterion 'regression(regressors/regr_for_oh.pkl)' \
  --net resnet50 \
  --da alda \
  --num-iterations 24 \
  --min-budget 2000 \
  --max-budget 6000 \
  --kill-diverging \
  --data-root /path/to/data
```

The script `launch_slurm_stub.sh` needs to be customized according to your cluster setup. A similar script can be developed for other schedulers, like PBS.
Once the job is done, a `result.pkl` file will be produced. To analyze the results, run

```
./analysis.py --result experiments/your-experiment/results_file.pkl
```

You can test a specific configuration with

```
./train_model.py \
    --net resnet50 \
    --da alda \
    --gpu 0 \
    --source art-oh \
    --target clipart-oh \
    --config base.weight_da=0.88,disc.dropout=0.1,disc.hidden_size_log=10,disc.num_fc_layers=5,net.bottleneck_size_log=9 \
    --data-root /path/to/data
```

### Contributors

* Luca Robbiano <luca.robbiano@polito.it>
* Muhammad Rameez Ur Rahman <rahman@di.uniroma1.it>
* Fabio Maria Carlucci

### License
This code and the attached files are distributed under the **BSD 3-Clause license**.
