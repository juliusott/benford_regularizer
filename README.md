# Benford Regularizer
This is the official implementation of the Benford Regularizer with quantile regression.

The following sections describe how to execute the code for the individual experiments, presented in the paper.


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

- ```-- epochs(Optional, default:250``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility between 0 and 1000000

- ```-- early_stop_patience(Optional, default:15)``` : Patience for Benford optimization

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- resume (Optional, default:False)``` : Training from checkpoint

- ```-- finetune (Optional, default:False)``` : Training from best model starting with benford iteration

- ```-- scale (Optional, default:1)``` : Scaling the Benford Optimization 

- ```-- benford_iter (Optional, default:10)``` : Positive number of iterations for the Benford regularization




### Example command for transfer learning experiments from ImageNet to Cifar10
```python
python run_experiments_transfer_learning.py --model densenet121 --epochs 50 --seed 150195

python run_experiments_transfer_learning.py --model densenet121 --epochs 50 --seed 150195 --benford --scale 0.1

```

### arguments

- ```-- model ``` : 'resnet50', 'resnet34', 'resnet18'

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:250``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- early_stop_patience(Optional, default:10)``` : Patience for Benford optimization

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- scale (Optional, default:1)``` : Scaling the Benford Optimization

- ```-- benford_iter (Optional, default:10)``` : Number of iterations for the Benford regularization

### Example command for training on Google Speech Commands
```python
python train_speech_commands.py --epochs 50 --seed 150195

```

### arguments

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:80``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- early_stop_patience(Optional, default:10)``` : Patience for Benford optimization

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- scale (Optional, default:1)``` : Scaling the Benford Optimization

- ```-- benford_iter (Optional, default:10)``` : Number of iterations for the Benford regularization

### Example command for training on the IRIS dataset
```python
python train_iris.py --epochs 50 --seed 150195

```

### arguments

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:50``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility

- ```-- early_stop_patience(Optional, default:100ß)``` : Patience for Benford optimization

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- scale (Optional, default:1)``` : Scaling the Benford Optimization 

- ```-- benford_iter (Optional, default:10)``` : Number of iterations for the Benford regularization
