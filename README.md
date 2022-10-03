# Unsupervised Deep Robust Non-Rigid Alignment by LowRank Los and Multi Input Attention

## Requirements
Python3.6
PyTorch
Hydra (hydra-core)

## Train & Validation 
You can train and validation by run.py
### Example
```
$ python run.py
```

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
    │   ├── erase                          # Erase is denoised input imgs used for test. Not used for training.
    │   │    ├── 0000.pt
    │   │    ├── 0001.pt
    │   │    ├── :
    │   │    └── n.pt
    │   └── transe                         # Transe is denoised & sparse complement input imgs used for test. Not used for training.
    │       ├── 0000.pt
    │       ├── 0001.pt
    │       ├── :
    │       └── n.pt
    │
    ├── eval_imgs
    │       (Same structure of train_imgs. Without eval_img, part of train_img is used for evaluation)
    └── test_imgs
            (Same structure of train_imgs)
```
</div></details>

### Arguments
You can set up input path/output path/parameters from 
[config/config.yaml](https://github.com/asanomitakanori/Unsupervised-Deep-Non-Rigid-Alignment-by-Low-Rank-Loss-and-Multi-Input-Attention/blob/main/config/config.yaml)

## Testing (measuring Dice scores)
To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:
```
$ python test.py
```

## Tensorboard 
command in teminal at working dir
```
tensorboard --logdir=./logs
```
## Citaiton
If you find the code useful for your research, please cite:
```
@inproceedings{asanomi2022unsupervised,
  title={Unsupervised Deep Non-rigid Alignment by Low-Rank Loss and Multi-input Attention},
  author={Asanomi, Takanori and Nishimura, Kazuya and Song, Heon and Hayashida, Junya and Sekiguchi, Hiroyuki and Yagi, Takayuki and Sato, Imari and Bise, Ryoma},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={185--195},
  year={2022},
  organization={Springer}
}
```
