## End-to-End Object Detection using Transformers, in `jax`.

This is a minimal implementation of DETR in `jax`, borrowing most details from 
[google-research/scenic](https://github.com/google-research/scenic/) and 
[google-research/big_vision](https://github.com/google-research/big_vision/).

What differs:
* Supports Sinkhorn solver based on latest OTT package (at the time of writing).
* Supports BigTransfer (BiT-S) ResNet-50 finetuned model on COCO.
* Knobs for training 