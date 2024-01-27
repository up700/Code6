## Requirements
- python==3.7.4
- pytorch==1.11.0
- [huggingface transformers](https://github.com/huggingface/transformers)
- numpy
- tqdm

## Overview
```
├── root
│   └── dataset
│       ├── 100 (K=100)
│       │   ├── augen100_train.json (The generated data by LLM)
│       │   ├── ...
│       │   ├── en100_train.json (The original low-resource data)
│       │   └── ...
│       ├── 200 (K=200)
│       │   └── ...
│       ├── 500 (K=500)
│       │   └── ...
│       ├── 1000 (K=1000)
│       │   └── ...
│       ├── en_dev.json
│       ├── en_test.json
│       ├── en_tag_to_id.json
│       └── ...
│   └── models
│       ├── __init__.py
│       └── modeling_decom.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── data_utils.py
│       ├── eval.py
│       └── loss_utils.py
│   └── run_script.py
│   └── run_data_augmentation.sh
```

## How to run
```console
sh run_data_augmentation.sh <GPU ID> <DATASET> <SAMPLES>
```
For example, English (En)
```console
sh run_bash.sh 0 En 1000
```
