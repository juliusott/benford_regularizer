# Benford Regularizer
This is the official implementation of the Benford Regularizer with quantile regression.

The following sections describe how to execute the code for the individual experiments, presented in the paper.


The code is tested with

* Python (3.9.16)
* Python (3.10.4)

## Code execution from terminal

### Example command for Cifar10 experiments
```python
python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --data_size 0.9 

python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --benford --data_size 0.9

```

### Example command for Cifar10 Finetuning with Benford experiments

```python
python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --resume --benford 
python run_experiments.py --model densenet121 --epochs 250 --seed 150195 --resume --benford --finetune

```
### arguments

- ```-- model ``` : 'PreActresnet101', 'PreActresnet50', 'densenet121', 'densenet169', 'renext', 'vitb16'

- ```-- lr (Optional, default:0.1)``` : inital learning rate

- ```-- epochs(Optional, default:250``` : number of epochs to train

- ```-- seed (Optional, default:42)``` : random seed for reproducibility between 0 and 1000000

- ```-- exclude_bias(Optional, default:False)``` : Excluding bias terms for Benford optimization

- ```-- scale(Optional, default:0.0001)``` : scaling factor for quantile regression loss

- ```-- benford(Optional, default:False)``` : Boolean to if Benford optimization should be used

- ```-- resume (Optional, default:False)``` : Training from checkpoint

- ```-- finetune (Optional, default:False)``` : Training from best model starting with benford iteration

- ```-- benford_iter (Optional, default:10)``` : Positive number of the epoch when benford regularization starts

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

- ```-- benford_iter (Optional, default:-1)``` : Number of iterations before the Benford regularization

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

- ```-- benford_iter (Optional, default:-1)``` : Number of iterations before the Benford regularization

-  ```-- data_size (Optional, default:1.0)``` : relative training dataset size
