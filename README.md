CS 4375 — Assignment 3

### Overview

This project is the implementation of a feed-forward neural network using Python and NumPy. The neural network compares the tanh, sigmoid, and ReLU activation functions, different network architectures, learning rates, optimizations(SGD vs Adam).

### Dataset

UCI Bank Marketing Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains 45,211 instances and 16 original features. 
Objective: is to predict whether a customer subscribes to a bank term deposit (yes/no).

Data was split into a 70% training, 15% avlidation, and 15% testing data.

### Installation and Execution

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full experiment:

```bash
python main.py --epochs 40 --batch-size 128 --seed 42 --learning-rates 0.001 0.01 --hidden-one 32 --hidden-two 64 32
```

The implementation uses vectorized forward propagation and backpropagation, sigmoid output activation with binary cross-entropy, and Xavier or He weight initialization.

### Experimental Setup and Results

A total of 24 configurations were tested using three activation functions, two learning rates (0.001 and 0.01), two architectures (32 and 64–32 neurons), and two optimizers (SGD and Adam). All configurations used 40 epochs, a batch size of 128, and random seed 42.

The best configuration for each activation was selected using validation macro-F1.

| Activation | Architecture | Optimizer | Learning rate | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|
| Sigmoid | 64–32 | Adam | 0.01 | 89.59% | 0.7348 |
| Tanh | 32 | Adam | 0.01 | 89.53% | 0.7304 |
| ReLU | 32 | Adam | 0.001 | 90.39% | 0.7367 |
| Logistic regression | — | — | — | 90.15% | 0.6996 |

Complete experiment results are saved in `results/experiments.csv`, with learning curves saved as PNG files.

### Questions

a.
Which activation function performed best, and why? Relate your answer to the learning curves and Q1.3.**

ReLU had the best performance. It achieved a test accuracy of 90.39% and a macro-F1 of 0.7367. Its validation loss stabilized around 0.205, while sigmoid and tanh showed larger increases in validation loss during later epochs. Also, ReLU helps to mitigate vanishing gradients due to its derivative being 1 for positive inputs. THis is unlike sigmoid, whose derivative approaches zero for large positive or negative inputs.

b.
How did your optimizer change convergence speed and final performance compared with plain SGD?**

Generally speaking, Adam performed better than SGD, achieving higher final validation macro-F1 in all 12 matched comparisons. For example, with thecombination of ReLU, a 0.001 learning rate, and 32 hidden neurons, Adam achieved 0.7502 validation macro-F1 compared with SGD's 0.6086. Adam's adaptive learning rates and momentum-like estimates can accelerate convergence. On a side note though, exact convergence speed was not separately measured.

c.
Is there evidence of overfitting in any configuration? What would you do about it?**

Yes, there is evidence. The actiation functions of Sigmoid and tanh showed clear overfitting. With these two, while validation loss increased, training loss continued decreasing. For the selected sigmoid model, final training loss was approximately 0.135 compared with validation loss of 0.246. ReLU showed milder overfitting. However, there are things we could possbly do. Some improvements may include L2 regularization, early stopping, reducing network complexity, or decreasing the # of epochs.

d.
How does your best network compare with the logistic-regression baseline? What does this say about your dataset?**

The best ReLU network achieved 90.39% test accuracy, which is a little bid higher than that logistic regression's 90.15%. ReLu's macro-F1 was also higher (0.7367 vs 0.6996), suggesting improved classification performance across the imbalanced classes. However, the relatively small accuracy difference suggests that logistic regression is already a strong baseline and that the additional network complexity provides only modest improvements.

### Assumptions

- Binary classification with a prediction threshold of 0.5.
- Random seed 42 for reproducibility.
- Macro-F1 used because the dataset is imbalanced.
- Validation data used for model selection; test data reserved for final evaluation.
- All experiments use the same dataset splits.
