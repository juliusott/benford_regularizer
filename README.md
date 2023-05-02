# Benford Regularizer
Implementation of the Benford Regularizer with quantile regression.

Execute Experiments:

CIFAR 10 classification from scratch:

The code is tested with

* Python (3.9.16)
* Python (3.10.4)
## Code execution from terminal

### Example command for Cifar10 experiments
```python
python run_experiments.py --model densenet121 --epochs 250 --seed 150195

python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --benford --scale 0.1

```

### Example command for Cifar10 Finetuning with Benford experiments

```python
python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --resume --benford --scale 0.1
python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --resume --benford --scale 0.1 --finetune

```
### arguments

- ```-- model ``` : 'PreActresnet101', 'PreActresnet50', 'densenet121', 'densenet169', 'renext'

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:250``` : implemented algorithms are sac, ddpg, td3

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- early_stop_patience(Optional, default:15)``` : Patience for Benford optimization

- ```-- benford(Optional, default:False)``` : If Benford optimization should be used

- ```-- resume (Optional, default:False)``` : Training from checkpoint

- ```-- finetune (Optional, default:False)``` : Training from best model starting with benford iteration

- ```-- scale (Optional, default:1)``` : Scaling the Benford Optimization if activated




### Example command for transfer learning experiments from ImageNet to Cifar10
```python
python run_experiments_transfer_learning.py --model densenet121 --epochs 50 --seed 150195

python run_experiments_transfer_learning.py --model densenet121 --epochs 50 --seed 150195 --benford --scale 0.1

```

### arguments

- ```-- model ``` : 'resnet50', 'resnet34', 'resnet18'

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:50``` : implemented algorithms are sac, ddpg, td3

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- early_stop_patience(Optional, default:5)``` : Patience for Benford optimization

- ```-- benford(Optional, default:False)``` : If Benford optimization should be used

- ```-- scale (Optional, default:1)``` : Scaling the Benford Optimization if activated ,00 00 
