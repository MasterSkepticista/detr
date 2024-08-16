# DETR: End-to-End Object Detection with Transformers.

This is a minimal implementation of [DETR](https://arxiv.org/abs/2005.12872) using `jax` and `flax`.

<table align="center">
  <tr>
    <td rowspan="2" align="center">
      <img src="https://github.com/MasterSkepticista/detr/raw/main/.github/detr.png" alt="DETR Architecture">
    </td>
    <td align="center" style="padding-bottom: 10px;">
      <img src="https://github.com/MasterSkepticista/detr/raw/main/.github/jax.png" alt="JAX logo" width="50%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/MasterSkepticista/detr/raw/main/.github/flax.png" alt="Flax logo" width="40%">
    </td>
  </tr>
</table>

Features:
* Supports Flash Attention (up to 50% faster over and above the following optimizations).
* Supports Sinkhorn solver (faster training for roughly the same final AP).
* Parallel bipartite matching for all auxiliary outputs (up to 30% faster training using Hungarian matcher).
* Uses `optax` API.
* Bug fixes from scenic to match official DETR implementation.
* Supports BigTransfer (BiT-S) ResNet-50 backbone.

You can read more about these optimizations [here](https://masterskepticista.github.io/portfolio/detr).

### Installation

* Setup:
  ```shell
  git clone https://github.com/MasterSkepticista/detr.git && cd detr
  python3.10 -m venv venv && source venv/bin/activate
  pip install -U pip setuptools wheel
  pip install -r requirements.txt
  ```

* You may need to download MS-COCO dataset in TFDS. Run the following to download
and create TFRecords:
  ```shell
  python -c "import tensorflow_datasets as tfds; tfds.load('coco/2017')"
  ```

### Train

Download torch resnet50 checkpoint from GDrive.

```shell
pip install gdown

# gdown <gdrive-file-id> -O <output-dir>
gdown 1q-PYc6ZshX12Nelb30V6Cp1FkmxhUdD2 -O artifacts/
```

|Backbone|Top-1 Acc.|Checkpoint|
|--------|----------|----|
|BiT-R50x1-i1k|76.8%|[Link](https://drive.google.com/file/d/1iVBV9jghBR2mseSc5z2SB1b8QptI9mju/view?usp=drive_link)|
|R50x1-i1k (from torchvision)|76.1%|[Link](https://drive.google.com/file/d/1q-PYc6ZshX12Nelb30V6Cp1FkmxhUdD2/view?usp=sharing) (created using this [gist](https://gist.github.com/MasterSkepticista/c854bce837a5cb5ca0489bd33b3a2259))|

```shell
# Trains the default DETR-R50-1333 model using `float32` precision.
# ~4.5 days on 8x A6000.
python main.py \
   --config configs/hungarian.py --workdir artifacts/`date '+%m-%d_%H%M'`
```

### Evaluate
Checkpoints (all non-DC5 variants) using the torchvision R50 backbone:

|Checkpoint|GFLOPs|$AP$|$AP_{50}$|$AP_{75}$|$AP_S$|$AP_M$|$AP_L$|
|-|-|-|-|-|-|-|-|
[DETR-R50-1333*](https://drive.google.com/file/d/1fu4M3l88mhiQEUpADoUT2wrSEIZNDSqe/view?usp=sharing)|174.2|40.80|61.88|42.45|19.2|44.31|60.32|
[DETR-R50-640](https://drive.google.com/file/d/1XYV3ULIDwa59AVYSAvBeIOFXwRR_GZ46/view?usp=sharing)|38.5|33.14|52.89|34.00|10.54|35.10|55.53|

\*official DETR baseline, except that these models were trained for 300 epochs instead of 500 epochs.

1. Download one of the pretrained checkpoints.
    ```python
    # In configs/common.py (or any)
    config.init_from = ml_collections.ConfigDict()
    config.init_from.checkpoint_path = '/path/to/checkpoint'
    ```
2. Replace `config.total_epochs` with `config.total_steps = 0` to skip to eval.

### Acknowledgements
Parts of this codebase are based on [scenic](https://github.com/google-research/scenic/).

DETR implementation in PyTorch: [facebookresearch/detr](https://github.com/facebookresearch/detr).

### License
MIT
