- Selezione TaskIL, DomainIL, ClassIL: variabile `mode` di tipo `settings.Mode`

# SGD

`python mnist_sgd.py`

## Split MNIST

- TaskIL: 91%
- ClassIL: 19%


## Permuted MNIST

- DomainIL: 46%

# EWC

`python ewc.py`

## Split MNIST

- TaskIL: 97%
- ClassIL: 20%


## Permuted MNIST

- DomainIL: 57%

# ER

`python er.py`

## Split MNIST

- TaskIL: 98% (Buffer 200) | 98% (Buffer 500) | 99% (Buffer 5120)
- ClassIL: 82% (Buffer 200) | 87% (Buffer 500) | 94% (Buffer 5120)

## Permuted MNIST

- DomaniIL: 76% (Buffer 200) | 83% (Buffer 500) | 90% (Buffer 5120)