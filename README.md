# MobileNetV3 TensorFlow
Unofficial implementation of MobileNetV3 architecture described in paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).
This repository follows [mobilenetv3-tensorflow](https://github.com/Bisonai/mobilenetv3-tensorflow).

## Requirements
* Python 3.6+
* TensorFlow 1.13+

```shell
pip install -r requirements.txt
```


## Build model

### MobileNetV3 Small
```python
from mobilenetv3_factory import build_mobilenetv3
model = build_mobilenetv3(
    "small",
    input_shape=(224, 224, 3),
    num_classes=3,
    width_multiplier=1.0,
)
```

### MobileNetV3 Large

```python
from mobilenetv3_factory import build_mobilenetv3
model = build_mobilenetv3(
    "large",
    input_shape=(224, 224, 3),
    num_classes=3,
    width_multiplier=1.0,
)
```

## Train

### Printer dataset

```shell
python train.py \
    --model_type small \
    --width_multiplier 1.0 \
    --height 224 \
    --width 224 \
    --dataset printer \
    --lr 0.001 \
    --optimizer sgd \
    --train_batch_size 10 \
    --valid_batch_size 10 \
    --num_epoch 30 \
    --logdir logdir
```


## Test

### printer dataset

```shell
python test.py \
	--model_path mobilenetv3_small_printer_30.h5
    --model_type small \
    --image_path test/1.jpg
```


## TensorBoard
Graph, training and evaluaion metrics are saved to TensorBoard event file uder directory specified with --logdir` argument during training.
You can launch TensorBoard using following command.

```shell
tensorboard --logdir logdir
```

