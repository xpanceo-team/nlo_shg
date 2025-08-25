# Nonlinear optics: ML model to predict Second harmonic generation (SHG)


## Install

Install python=3.10 or higher
To install dependencies run:

**1.** Install torch==2.1.2:

```pip install torch==2.1.2```

**2.**
Install torch-scatter and torch-sparse. For example for CUDA 12.1:

```pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html```

```pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html```

**3.** Install other dependencies:

```pip install -r requirements.txt```

## Infer model

### Data format.
Data samples to infer have to be in following format:

```
[
    {
        'atoms': <dict from jarvis.core.atoms.Atoms object>,
        'shg': np.zeros(3, 6),
        'crystal_class': 'point group symbol'
    },
    ...
]
```

Data must be save in pickle format (see examples in ```data/samples```)

### Inference
To infer trained model run:

```bash
cd shg
python infer.py \
--raw_data ../data/samples/infer_data.pickle \
--path2save_data ../data/samples/preprocessed_infer_data.pickle \
--load_model True \
--model_path ../data/weights/cluster_noemd_optixnet.pt \
--output_path output_infer.json \
```

This will run model and save inference results in output_infer.json file.

## Training model

You also can find training script that authors used to train OptiXNet (```shg/train.py```).

In case you wish to train your own model please prepare your data in format described above (note, that you have to replace 'shg' to calculated shg tensor with shape (3, 6)).

Run training with following command:

```
python train.py \
--raw_data ../data/samples/train_data.pickle\
--path2save_data ../data/samples/preprocessed_train_data.pickle \
--name train_optixnet_on_custom_dataset \
--target shg \
--epochs 100 \
--batch_size 32 \
--learning_rate 0.001 \
--loss l1 \
--weight_decay 0.001
```

This will run training model. Trained model will be saved in ```runs/train_optixnet_on_custom_dataset/``` directory


Please feel free to change training script and preparing data in any way. 

## Citiation:
**TODO**

Thanks to [AIRS/OpenMat](https://github.com/divelab/AIRS/tree/main/OpenMat/GMTNet) repository to GMTNet implementation. Our code was based on this implementation and modified.
