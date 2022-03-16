# Leveraging Unimodal Self-Supervised Learning for Multimodal Audio-Visual Speech Recognition
Official PyTorch implementation of paper Leveraging Uni-Modal Self Supervised Learning for Multimodal Audio-visual Speech Recognition
## How to install environment
1. Clone the repo into a directory. 
```shell
git clone https://github.com/LUMIA-Group/Leveraging-Self-Supervised-Learning-for-AVSR.git
```
2. Install all required packages.
```shell
pip install -r requirements.txt
```
Noted that the Pytorch-Lightning lib do not support a wrapped ReduceLROnPlateau scheduler, we need to modify the lib manually by:
```shell
python -c "exec(\"import pytorch_lightning\nprint(pytorch_lightning.__file__)\")"
vi /path/to/pytorch_lightning/trainer/optimizers.py
```
and comments the 154-156 lines
```python
# scheduler["reduce_on_plateau"] = isinstance(
#     scheduler["scheduler"], optim.lr_scheduler.ReduceLROnPlateau
# )
```
## How to prepare dataset
1. Download [LRW dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) and [LRS2 dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
2. Download pretrained [MoCo v2 model](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar) and [wav2vec 2.0 model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt)
3. Change the directory in `config.py` to "relative directory" relative to the project root directory
4. Preprocessing the LRW dataset.
```shell
cd trainFrontend
python saveh5.py
```
5. Preprocessing the LRS2 dataset.
```shell
python saveh5.py
```

## How to train
1. Train the visual front-end on LRW.
```shell
python trainfrontend.py
```
2. Change the `args["MOCO_FRONTEND_FILE"]` in `config.py` to the trained front-end file, and config `args["MODAL"]` to choose modality.
3. Train the AO and VO model first.
```shell
python train.py
```
4. Then train the AV model. Before that, change the `args["TRAINED_AO_FILE"]` and `args["TRAINED_VO_FILE"]` to the trained AO and VO model.
```shell
python train.py
```
## How to test
1. Choose test configuration and model.
2. Evaluate the visual word classification performance.
```shell
python evalfrontend.py
```
3. Evaluate the AO/VO/AV model.
```shell
python eval.py
```
