
import ssl
import certifi

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# storin URL fr the UCI Bank Marketing dataset
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


# downloading & loading UCI Bank Marketing dataset.
def load_bank_data(cache_dir):

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive = cache_dir / "bank_marketing.zip"

    if not archive.exists():

        print("Downloading UCI Bank Marketing dataset...")
        

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATA_URL, context=context) as response:
            archive.write_bytes(response.read())

    with zipfile.ZipFile(archive) as outer:

        if "bank-full.csv" in outer.namelist():

            with outer.open("bank-full.csv") as f:
                df = pd.read_csv(f, sep=";")

        elif "bank.zip" in outer.namelist():

            from io import BytesIO

            with zipfile.ZipFile(BytesIO(outer.read("bank.zip"))) as inner:

                with inner.open("bank-full.csv") as f:
                    df = pd.read_csv(f, sep=";")
        else:

            raise ValueError("UCI archive did not contain bank-full.csv")

    X = df.drop(columns=["y"])
    y = (df["y"] == "yes").astype(int).to_numpy()

    return X, y


# implementin Neural network using NumPy
# supporting different activation functions, architectures, and optimizers
class ScratchNeuralNetwork:

    # fully connected neural network with manual backpropagation.

    def __init__(
            
        self,
        input_dim,
        hidden,
        activation,
        optimizer,
        lr,
        seed,
        batch_size=128,
        epochs=40,
        beta=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        
        self.hidden = tuple(hidden)
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.beta = beta
        self.beta2 = beta2
        self.eps = eps

        self.rng = np.random.default_rng(seed)

        # initializin weights and biases for each layer
        # using He initialization for ReLU and Xavier for sigmoid and tanh
        sizes = [input_dim, *hidden, 1]

        self.W = []
        self.b = []

        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):

            if activation == "relu":
                std = np.sqrt(2 / fan_in)

            else:
                std = np.sqrt(2 / (fan_in + fan_out))

            self.W.append(
                self.rng.normal(0, std, (fan_in, fan_out))
            )

            self.b.append(np.zeros((1, fan_out)))

        # initializing variables used by the Adam optimizer
        self.step = 0

        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    @staticmethod
    def sigmoid(z):

        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # apploying the selected hidden-layer activation function
    def activate(self, z):

        if self.activation == "sigmoid":
            return self.sigmoid(z)

        if self.activation == "tanh":
            return np.tanh(z)

        return np.maximum(0, z)

    # calculating activation derivatives for backpropagation
    def derivative(self, z):

        if self.activation == "sigmoid":
            s = self.sigmoid(z)
            return s * (1 - s)


        if self.activation == "tanh":
            return 1 - np.tanh(z) ** 2


        return (z > 0).astype(float)

    # showing vectorized forward propagation.
    def forward(self, X):

        activations = [X]
        preactivations = []

        for W, b in zip(self.W[:-1], self.b[:-1]):

            z = activations[-1] @ W + b

            preactivations.append(z)
            activations.append(self.activate(z))

        logits = activations[-1] @ self.W[-1] + self.b[-1]

        preactivations.append(logits)
        activations.append(self.sigmoid(logits))

        return activations, preactivations

    # showing vectorized backpropagation.
    def backward(self, y, activations, preactivations):

        m = len(y)

        delta = (
            activations[-1] - y.reshape(-1, 1)
        ) / m

        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        for i in range(len(self.W) - 1, -1, -1):

            dW[i] = activations[i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:

                delta = (
                    delta @ self.W[i].T
                ) * self.derivative(preactivations[i - 1])

        return dW, db

    # updating weights using SGD or Adam.
    def update(self, dW, db):

        self.step += 1

        for i in range(len(self.W)):

            if self.optimizer == "sgd":
                
                self.W[i] -= self.lr * dW[i]
                self.b[i] -= self.lr * db[i]

            else:

                self.mW[i] = (

                    self.beta * self.mW[i]
                    + (1 - self.beta) * dW[i]
                )

                self.vW[i] = (

                    self.beta2 * self.vW[i]
                    + (1 - self.beta2) * dW[i] ** 2
                )

                self.mb[i] = (

                    self.beta * self.mb[i]
                    + (1 - self.beta) * db[i]
                )

                self.vb[i] = (

                    self.beta2 * self.vb[i]
                    + (1 - self.beta2) * db[i] ** 2
                )

                mw = self.mW[i] / (1 - self.beta ** self.step)
                vw = self.vW[i] / (1 - self.beta2 ** self.step)

                mb = self.mb[i] / (1 - self.beta ** self.step)
                vb = self.vb[i] / (1 - self.beta2 ** self.step)

                self.W[i] -= (
                    self.lr * mw / (np.sqrt(vw) + self.eps)
                )


                self.b[i] -= (
                    self.lr * mb / (np.sqrt(vb) + self.eps)
                )

    def predict_proba(self, X):
        return self.forward(X)[0][-1].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    # calculating binary cross-entropy loss, accuracy, and macro-F1
    def evaluate(self, X, y):

        p = np.clip(

            self.predict_proba(X),
            1e-12,
            1 - 1e-12
        )

        loss = -np.mean(

            y * np.log(p)
            + (1 - y) * np.log(1 - p)
        )

        pred = (p >= 0.5).astype(int)

        return {

            "loss": float(loss),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(
                f1_score(

                    y,
                    pred,
                    average="macro",
                    zero_division=0
                )
            ),
        }

    # training with shuffled mini-batches.
    def train(self, X_train, y_train, X_val, y_val):

        history = {
            "train_loss": [],
            "val_loss": []
        }

        for _ in range(self.epochs):

            order = self.rng.permutation(len(y_train))

            for start in range(

                0,
                len(order),
                self.batch_size
            ):
                indices = order[
                    start:start + self.batch_size
                ]

                a, z = self.forward(X_train[indices])

                dW, db = self.backward(

                    y_train[indices],
                    a,
                    z
                )

                self.update(dW, db)

            history["train_loss"].append(
                self.evaluate(X_train, y_train)["loss"]
            )

            history["val_loss"].append(
                self.evaluate(X_val, y_val)["loss"]
            )

        return history


class AssignmentExperiment:
    #preprocessing dataset & experiment management.

    def __init__(self, args):

        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)

    # dividing dataset and preparing numerical/categorical features
    # fitting preprocessing transformations only on the training data
    def preprocess(self):

        X, y = load_bank_data(self.args.cache)

        X_train, X_temp, y_train, y_temp = train_test_split(

            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=self.args.seed
        )

        X_val, X_test, y_val, y_test = train_test_split(

            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=self.args.seed
        )

        numeric = X.select_dtypes(

            include=np.number
        ).columns.tolist()

        categorical = X.select_dtypes(

            exclude=np.number
        ).columns.tolist()

        num_pipe = Pipeline([

            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ])

        cat_pipe = Pipeline([

            (
                "impute",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transform = ColumnTransformer(
            [
                ("numeric", num_pipe, numeric),
                ("categorical", cat_pipe, categorical)
            ],
            sparse_threshold=0
        )

        train = np.asarray(

            transform.fit_transform(X_train),
            dtype=np.float64
        )

        val = np.asarray(

            transform.transform(X_val),
            dtype=np.float64
        )

        test = np.asarray(

            transform.transform(X_test),
            dtype=np.float64
        )

        print(

            f"Data split: train={len(y_train)}, "
            f"validation={len(y_val)}, "
            f"test={len(y_test)}; "
            f"features={train.shape[1]}, "
            f"positive rate={y.mean():.3f}"
        )

        return train, val, test, y_train, y_val, y_test

    def run(self):
        
        Xtr, Xv, Xt, ytr, yv, yt = self.preprocess()

        # baseline for Logistic regression
        baseline = LogisticRegression(

            max_iter=self.args.baseline_max_iter,
            random_state=self.args.seed
        )

        baseline.fit(Xtr, ytr)
        baseline_pred = baseline.predict(Xt)

        baseline_metrics = {

            "test_accuracy": float(
                accuracy_score(yt, baseline_pred)
            ),

            "test_macro_f1": float(
                f1_score(
                    yt,
                    baseline_pred,
                    average="macro"
                )
            )
        }

        print(

            "Logistic regression baseline:",
            baseline_metrics
        )

        # storing experiment results and tracking best model for each activation
        rows = []
        best = {}

        # 3 activations * 2 learning rates * 2 architectures * 2 optimizers = 24 runs
        
        for activation in ["sigmoid", "tanh", "relu"]:

            for lr in self.args.learning_rates:

                for hidden in [

                    self.args.hidden_one,
                    self.args.hidden_two
                ]:
                    for optimizer in ["sgd", "adam"]:

                        net = ScratchNeuralNetwork(

                            input_dim=Xtr.shape[1],
                            hidden=hidden,
                            activation=activation,
                            optimizer=optimizer,
                            lr=lr,
                            seed=self.args.seed,
                            batch_size=self.args.batch_size,
                            epochs=self.args.epochs
                        )

                        history = net.train(

                            Xtr,
                            ytr,
                            Xv,
                            yv
                        )

                        tr = net.evaluate(Xtr, ytr)
                        va = net.evaluate(Xv, yv)

                        record = {

                            "activation": activation,
                            "learning_rate": lr,
                            "hidden": "-".join(
                                map(str, hidden)
                            ),
                            "optimizer": optimizer,
                            "epochs": self.args.epochs,
                            "batch_size": self.args.batch_size,
                            "train_accuracy": tr["accuracy"],
                            "validation_accuracy": va["accuracy"],
                            "train_macro_f1": tr["macro_f1"],
                            "validation_macro_f1": va["macro_f1"],
                            "final_train_loss": tr["loss"],
                            "final_val_loss": va["loss"]
                        }

                        rows.append(record)

                        print(

                            f"{len(rows):02d}/24 "
                            f"{activation:7s} "
                            f"{str(hidden):10s} "
                            f"{optimizer:4s} "
                            f"lr={lr:g} "
                            f"train={tr['accuracy']:.4f} "
                            f"val={va['accuracy']:.4f} "
                            f"val_macro_f1={va['macro_f1']:.4f}",
                            flush=True
                        )

                        # chossing best by validation macro-F1.
                        # Validation accuracy breaks ties.
                        score = (

                            va["macro_f1"],
                            va["accuracy"]
                        )

                        if (

                            activation not in best
                            or score > best[activation]["score"]
                        ):
                            best[activation] = {
                                "score": score,
                                "net": net,
                                "history": history,
                                "config": record.copy()
                            }

        # saing all experiment results
        table = pd.DataFrame(rows)

        table.to_csv(

            self.output / "experiments.csv",
            index=False
        )

        test_rows = []

        for activation, winner in best.items():

            # evaluating test data only after model selection
            metrics = winner["net"].evaluate(Xt, yt)

            row = {

                **winner["config"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_loss": metrics["loss"]
            }

            test_rows.append(row)

            # plotting learning curves
            hist = winner["history"]

            plt.figure(figsize=(8, 5))

            epochs = np.arange(

                1,
                len(hist["train_loss"]) + 1
            )

            plt.plot(

                epochs,
                hist["train_loss"],
                label="Training loss"
            )

            plt.plot(

                epochs,
                hist["val_loss"],
                label="Validation loss"
            )

            plt.xlabel("Epoch")
            plt.ylabel("Binary cross-entropy")

            plt.title(
                f"Best {activation} model: learning curves"
            )

            plt.legend()
            plt.tight_layout()

            plt.savefig(

                self.output
                / f"learning_curve_{activation}.png",
                dpi=160
            )

            plt.close()

        pd.DataFrame(test_rows).to_csv(

            self.output / "best_test_results.csv",
            index=False
        )

        with (
            self.output / "baseline.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)

        with (
            self.output / "run_config.json"
        ).open("w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2)

        print(

            "\nBest configuration per activation "
            "(test evaluated once per winner):"
        )

        print(

            pd.DataFrame(test_rows)[
                [
                    "activation",
                    "hidden",
                    "optimizer",
                    "learning_rate",
                    "validation_accuracy",
                    "validation_macro_f1",
                    "test_accuracy",
                    "test_macro_f1"
                ]
            ].to_string(index=False)
        )

        print(
            f"\nFiles saved in: {self.output.resolve()}"
        )


# defining command-line arguments (for reproducible experiments)
def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "NumPy neural network experiment "
            "(UCI Bank Marketing)"
        )
    )

    parser.add_argument(

        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(

        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(

        "--learning-rates",
        type=float,
        nargs=2,
        default=[0.001, 0.01]
    )

    parser.add_argument(

        "--hidden-one",
        type=int,
        nargs=1,
        default=[32]
    )

    parser.add_argument(

        "--hidden-two",
        type=int,
        nargs=2,
        default=[64, 32]
    )

    parser.add_argument(

        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(

        "--baseline-max-iter",
        type=int,
        default=2000
    )

    parser.add_argument(

        "--cache",
        default="data_cache"
    )

    parser.add_argument(

        "--output",
        default="results"
    )

    args = parser.parse_args()

    if (
        
        args.epochs < 1
        or args.batch_size < 1
        or any(
            n < 1
            for n in args.hidden_one + args.hidden_two
        )
    ):
        parser.error(
            "Epochs, batch size, and hidden-layer "
            "sizes must be positive."
        )

    if any(
        lr <= 0
        for lr in args.learning_rates
    ):
        parser.error(
            "Learning rates must be positive."
        )

    return args


if __name__ == "__main__":
    config = parse_args()

    print("Random seed:", config.seed)

    AssignmentExperiment(config).run()
