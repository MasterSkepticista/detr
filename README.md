# DETR: End-to-End Object Detection with Transformers.

This is a minimal implementation of [DETR](https://arxiv.org/abs/2005.12872) in `jax`.

What differs here, from the implementation in [scenic](https://github.com/google-research/scenic/):
* Supports Sinkhorn solver based on latest OTT package (at the time of writing).
* Supports BigTransfer (BiT-S) ResNet-50 backbone.

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

* Download and extract `instances_val2017.json` from [MS-COCO](https://cocodataset.org/#download) in the root directory of this repo.

### Checkpoints

Set the checkpoint path in the chosen config file.

|Backbone|Top-1 Acc.|Checkpoint|
|--------|----------|----|
|BiT-R50x1-i1k|76.8%|[Link](https://drive.google.com/file/d/1iVBV9jghBR2mseSc5z2SB1b8QptI9mju/view?usp=drive_link)|
|R50x1-i1k|76.1%|[Link](https://drive.google.com/file/d/14N0upIZHSlFkvF4E8NNH8dxKVwS6RQjb/view?usp=drive_link)|

_I will add full model checkpoints soon._

### Train
```shell
$> python main.py \
   --config configs/sinkhorn.py --workdir artifacts/`date '+%m-%d_%H%M'`
```

### Acknowledgements
Large parts of this codebase were motivated by implementations in [scenic](https://github.com/google-research/scenic/) and 
[big_vision](https://github.com/google-research/big_vision/).