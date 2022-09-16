# Unsupervised-Deep-Robust-Non-Rigid-Alignment-by-Low-Rank-Loss-and-Multi-Input-Attention

## Requirements
Python3.6
PyTorch
Hydra (hydra-core)

## Training
You can train and validation by run.py
### Example
```
$ python run.py
```

## Train 
### Preparation
Please prepare your data as follows

<details><summary>current dir</summary><div>

```
./data
    ├── train_imgs
    │   ├── img                            # Arbitrary input imgs
    │   │   ├── 0000.pt                    # 0000.pt has 8 imgs. 8 is batch_size.
    │   │   ├── 0001.pt
    │   │   ├── :
    │   │   └── n.pt
    │   ├── erase                          # This is denoised imgs used for test. Not used for training.
    │   │    ├── 0000.pt
    │   │    ├── 0001.pt
    │   │    ├── :
    │   │    └── n.pt
    │   └── transe                         # Transe is denoised & sparse complement imgs used for test. Not used for training.
    │       ├── 0000.pt
    │       ├── 0001.pt
    │       ├── :
    │       └── n.pt
    │
    ├── eval_imgs
    │       (Same structure of train_imgs. Without eval_img, part of train_img is used for evaluation)
    └── test_imgs
            (Same structure of train_imgs)
