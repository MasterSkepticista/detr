# DETR: End-to-End Object Detection with Transformers.

This is a minimal implementation of [DETR](https://arxiv.org/abs/2005.12872) using `jax` and `flax`.

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/MasterSkepticista/detr/raw/main/.github/detr.png" alt="DETR Architecture">
</div>
<div style="display: flex; justify-content: center; align-items: center; margin-top: 10px;">
  <img src="https://github.com/MasterSkepticista/detr/raw/main/.github/jax.png" alt="JAX logo" width="20%" style="margin-right: 30px;">
  <img src="https://github.com/MasterSkepticista/detr/raw/main/.github/flax.png" alt="JAX logo" width="12%">
</div>


### Getting Started

* Setup:
  ```shell
  $> git clone https://github.com/MasterSkepticista/detr.git
  $> cd detr/
  $> python3.10 -m venv venv
  $> source venv/bin/activate
  (venv) $> pip install -U pip setuptools wheel
  (venv) $> pip install -r requirements.txt
  ```

* You may need to download MS-COCO dataset in TFDS. Run the following to download
and create TFRecords:
  ```shell
  (venv) $> python -c "import tensorflow_datasets as tfds; tfds.load('coco/2017')"
  ```

* Download and extract `instances_val2017.json` from [MS-COCO](https://cocodataset.org/#download) 
in the root directory of this repo (or update `config.annotations_loc` in the config).

### Checkpoints

Place these checkpoints under a new directory `artifacts` at the root of this repository.
Alternatively, modify `config.pretrained_backbone_configs.checkpoint_path` in the chosen config file.

|Backbone|Top-1 Acc.|Checkpoint|
|--------|----------|----|
|BiT-R50x1-i1k|76.8%|[Link](https://drive.google.com/file/d/1iVBV9jghBR2mseSc5z2SB1b8QptI9mju/view?usp=drive_link)|
|R50x1-i1k|76.1%|[Link](https://drive.google.com/file/d/14N0upIZHSlFkvF4E8NNH8dxKVwS6RQjb/view?usp=drive_link)|

Checkpoints (all non-DC5 variants) created using this repository (all 300 epoch schedule):

|Checkpoint|GFLOPs|$AP$|$AP_{50}$|$AP_{75}$|$AP_S$|$AP_M$|$AP_L$|
|-|-|-|-|-|-|-|-|
[DETR-R50-640](https://drive.google.com/file/d/1XYV3ULIDwa59AVYSAvBeIOFXwRR_GZ46/view?usp=sharing)|38.5|33.14|52.89|34.00|10.54|35.10|55.53|
<!-- [DETR-R50-1333*]()|33.14|52.89|34.00|10.54|35.10|55.53| -->

\*matches official DETR baseline, except for 300ep instead of 500ep.

### Train
```shell
# Trains the default DETR-R50-1333 model.
$> python main.py \
   --config configs/hungarian.py --workdir artifacts/`date '+%m-%d_%H%M'`
```

### Evaluate
1. Download one of the pretrained checkpoints.
    ```python
    # In configs/common.py (or any)
    config.init_from = ml_collections.ConfigDict()
    config.init_from.checkpoint_path = '/path/to/checkpoint'
    ```
2. Replace `config.total_epochs` with `config.total_steps = 0` to skip to eval.

### Acknowledgements
Large parts of this codebase were motivated by [scenic](https://github.com/google-research/scenic/) and 
[big_vision](https://github.com/google-research/big_vision/).

What differs here, from the implementation in [scenic](https://github.com/google-research/scenic/):
* Supports Sinkhorn solver based on latest OTT package (at the time of writing).
* Supports BigTransfer (BiT-S) ResNet-50 backbone.
* Bug fixes to match official DETR implementation.

### Contributing
I maintain this project on a best-effort basis. Please raise an issue if you face
problems using this repository.

PR contributions are welcome.