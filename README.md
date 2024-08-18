# Benford Regularizer
This is the official implementation of the Benford Regularizer with quantile regression.

The following sections describe how to execute the code for the individual experiments, presented in the paper.


The code is tested with

* Python (3.9.16)
* Python (3.10.4)

## Code execution from terminal

### Example command for Cifar10 experiments or Cifar100
```python
python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --data_size 0.9 

python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --benford --dataset "cifar100"

```

### arguments

- ```-- model ``` : 'PreActresnet101', 'PreActresnet50', 'densenet121', 'densenet169', 'renext', 'vit', 'swin'

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:250``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility between 0 and 1000000

- ```-- scale(Optional, default:0.0001)``` : scaling factor for quantile regression loss

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- resume (Optional, default:False)``` : Training from checkpoint

- ```-- finetune (Optional, default:False)``` : Training from best model starting with benford iteration

-  ```-- data_size (Optional, default:1.0)``` : relative training dataset size

### Example command for Imagenet evaluation an training
```python
python train_imagenet.py --multiprocessing-distributed --dist-url "file:///home/benford_regularizer/temp_file" --rank 0 --world-size 1 --batch-size 256 --benford --scale 1e-6

python train_imagenet.py --resume "./experiments/imagenet_mobilenet_v3_small_benford_False_subset_1.0/ckpt_mobilenet_v3_small_9819323527973343954False.pth" --evaluate --model mobilenet_v3_small


```

### arguments
see https://github.com/pytorch/examples/tree/main/imagenet for more information about the distributed learning and the sepcific arguments

- ```-- model ``` : torchvision model

- ```-- multiprocessing-distributed ```: flag for distributed learning

- ```-- dist-url ```: url for distributed learning

- ```-- rank ```: node rank

- ```-- world-size ```: number of nodes

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:250``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility between 0 and 1000000

- ```-- scale(Optional, default:0.0001)``` : scaling factor for quantile regression loss

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- resume (Optional, default:False)``` : Training from checkpoint

- ```-- finetune (Optional, default:False)``` : Training from best model starting with benford iteration

-  ```-- data_size (Optional, default:1.0)``` : relative training dataset size


### Example command for training on Google Speech Commands
```python
python train_speech_commands.py --epochs 50 --seed 150195

```

### arguments

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:80``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- scale(Optional, default:0.01)``` : scaling factor for quantile regression loss

-  ```-- data_size (Optional, default:1.0)``` : relative training dataset size

### Example command for training on the IRIS dataset
```python
python train_iris.py --epochs 50 --seed 150195

```

### arguments

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:50``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- scale (Optional, default:1)``` : scaling factor for quantile regression loss 

-  ```-- data_size (Optional, default:1.0)``` : relative training dataset size
