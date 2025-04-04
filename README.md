# DGCL
Domain Generalizable Continual Learning

### Requirements

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)
7. [easydict](https://github.com/makinacorpus/easydict)
8. [matplotlib](https://github.com/matplotlib/matplotlib)


### Usage

#### Data Preparation

Download the dataset and then update `utils/data.py` with the path to your data directory.

#### Baseline Experiments

1. For implemented methods (listed in `models` folder), follow the bash scripts named `run_*.sh` to run the experiments.

2. For detailed configurations, please refer to the `configs` folder, baseline configs are located in `configs/DGIL/[dataset_name]`, where `[method_name].json` runs the regular class incremental learning pipeline, and `[method_name]_dgil.json` runs the domain generalizable continual learning pipeline.

#### DoT Experiments

Here specifically, we provide the scripts to run the DoT experiments on DigitsDG and OfficeHome datasets with SLCA and L2P:

1. DoT-SLCA:
```bash
# DoT-SLCA on DigitsDG
python main.py --config ./configs/DGIL/digitsdg/dot_slca_dgil.json

# DoT-SLCA on OfficeHome
python main.py --config ./configs/DGIL/officehome/dot_slca_dgil.json
```

2. DoT-L2P:
```bash
# DoT-L2P on DigitsDG
python main.py --config ./configs/DGIL/digitsdg/dot_l2p_dgil.json

# DoT-L2P on OfficeHome
python main.py --config ./configs/DGIL/officehome/dot_l2p_dgil.json
```