# Randomly Initialized Classification

Let's look at neural networks that are randomly initialized. Attach and train a classification head on the features produced by the randomly initialized networks.


### Untrained Cifar10 Performance

Linear classification head trained for 25 epochs using ADAM with learning rate 0.0001, weigth decay of 1e-5.


| Model | Loss |  top1 | top5  |
|-------|------|-------|-------|
| Resnet18 | 1.879467 | 33.06 | 82.20 |
| Resnet50 | 2.144986 | 22.36 | 71.68 |
| Resnet152 | 2.246238 | 16.03 | 63.43 |
| ViT | 1.921553| 31.82 | 82.48 |

### ImageNet Pretrained Cifar10 Performance

Linear classification head trained for 25 epochs using ADAM with learning rate 0.0001, weigth decay of 1e-5.


| Model | Loss |  top1 | top5  |
|-------|------|-------|-------|
| Resnet18 | 1.543931 | 47.31| 90.53 |
| Resnet50 | 1.347822 | 54.62 | 93.14 |
| Resnet152  | 2.185608 |  51.69 | 91.96 |

