from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from GCN.dataset import Dataset
from GCN import model as m
from model.callback import *
from datetime import datetime
import numpy as np
import time
import csv
import os


class Trainer(object):
    def __init__(self, dataset):
        self.data = None
        self.model = None
        self.hyper = {"dataset": dataset}
        self.log = {}

    def __repr__(self):
        text = ""
        for key, value in self.log.items():
            text += "{}:\t".format(key)
            for error in value[0]:
                text += "{0:.4f} ".format(float(error))
            text += "\n"

        return text

    def load_data(self, batch=128, max_number=0):
        self.data = Dataset(self.hyper["dataset"], batch=batch)
        self.hyper["num_train"] = len(self.data.y["train"])
        self.hyper["num_val"] = len(self.data.y["valid"])
        self.hyper["num_test"] = len(self.data.y["test"])
        self.hyper["num_atoms"] = self.data.num_atoms
        self.hyper["num_degrees"] = self.data.num_degrees
        self.hyper["num_features"] = self.data.num_atom_features
        self.hyper["data_std"] = self.data.std
        self.hyper["data_mean"] = self.data.mean
        self.hyper["task"] = self.data.task
        self.hyper["batch"] = batch
        self.hyper["max_number"] = max_number

    def load_model(self, model, units_conv=128, units_dense=128, num_layers=2, loss="mse"):
        self.hyper["model"] = model
        self.hyper["units_conv"] = units_conv
        self.hyper["units_dense"] = units_dense
        self.hyper["num_layers"] = num_layers
        self.hyper["loss"] = loss
        self.model = getattr(m, model)(self.hyper)
        self.model.summary()

    def fit(self, model, epoch, batch=128, fold=10, units_conv=128, units_dense=128, num_layers=2,
            loss="mse", monitor="val_rmse"):
        # 1. Generate CV folder
        now = datetime.now()
        base_path = "../../result/{}/{}/".format(model, self.hyper["dataset"])
        log_path = base_path
        results = []

        for i in range(1, fold + 1):
            start_time = time.time()

            # 2. Generate data
            self.load_data(batch=batch)

            # 3. Make model
            self.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers, loss=loss)

            # 4. Callbacks
            log_path = base_path + "{}_c{}_d{}_l{}_{}/".format(batch, units_conv, units_dense, num_layers,
                                                               now.strftime("%m%d%H"))
            tb_path = log_path + "trial_{}/".format(i)

            callbacks = []
            if self.data.task != "regression":
                callbacks.append(Roc(self.data.tensorize(target="valid", task=self.data.task), self.data.val_step))
                mode = "max"
            callbacks += [Tensorboard(log_dir=tb_path, write_graph=False, histogram_freq=0, write_images=True),
                          ModelCheckpoint(tb_path + "{epoch:01d}-{" + monitor + ":.3f}.hdf5", monitor=monitor,
                                          save_weights_only=True, save_best_only=True, period=1, mode=mode),
                          EarlyStopping(patience=15),
                          ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=5, min_lr=0.0005)]

            # 5. Fit
            self.model.fit_generator(self.data.tensorize(target="train", task=self.data.task), epochs=epoch,
                                     steps_per_epoch=self.data.step,
                                     validation_data=self.data.tensorize(target="valid"),
                                     validation_steps=self.data.val_step,
                                     callbacks=callbacks)
            self.hyper["train_time"] = time.time() - start_time

            # 6. Find best checkpoint
            models = []
            for root, dirs, files in os.walk(tb_path):
                for fname in files:
                    if "hdf5" in fname:
                        models.append([fname[:-5].split("-")[0], fname[:-5].split("-")[1]])
            if self.data.task == "regression":
                idx = np.argmin(np.array(models), axis=0)[-1]
            else:
                idx = np.argmax(np.array(models), axis=0)[-1]
            best_model = tb_path + str(models[idx][0]) + "-" + str(models[idx][1]) + ".hdf5"
            self.model.load_weights(best_model)

            # 7. Save train, valid, test losses
            if self.data.task == "regression":
                train_loss = self.model.evaluate_generator(self.data.tensorize(target="train", task=self.data.task),
                                                           steps=self.data.step)
                valid_loss = self.model.evaluate_generator(self.data.tensorize(target="valid", task=self.data.task),
                                                           steps=self.data.val_step)
                test_loss = self.model.evaluate_generator(self.data.tensorize(target="test", task=self.data.task),
                                                          steps=self.data.test_step)

                results.append([train_loss[1], valid_loss[1], test_loss[1], train_loss[2], valid_loss[2], test_loss[2]])

            else:
                losses = []
                for gen, step in zip([self.data.tensorize(target="train", task=self.data.task),
                                      self.data.tensorize(target="valid", task=self.data.task),
                                      self.data.tensorize(target="test", task=self.data.task)],
                                     [self.data.step, self.data.val_step, self.data.test_step]):
                    y_true = []
                    y_pred = []
                    for _ in range(step):
                        x, y = next(gen)
                        y_true = np.concatenate([y_true, y])
                        y_pred = np.concatenate([y_pred, self.model.predict(x).flatten()])

                    losses.append(roc_auc_score(y_true, y_pred))
                    losses.append(average_precision_score(y_true, y_pred))

                results.append([losses[0], losses[2], losses[4], losses[1], losses[3], losses[5]])

            # 8. Save hyper
            with open(tb_path + "hyper.csv", "w") as file:
                writer = csv.DictWriter(file, fieldnames=list(self.hyper.keys()))
                writer.writeheader()
                writer.writerow(self.hyper)

            # 9. Save test results
            pred = self.model.predict_generator(self.data.tensorize(target="test", task="input_only"),
                                                steps=self.data.test_step)
            self.data.save_dataset(pred, tb_path, target="test")

            # Save cross validation results
        if self.data.task == "regression":
            header = ["train_mae", "valid_mae", "test_mae", "train_rmse", "valid_rmse", "test_rmse"]
        else:
            header = ["train_roc", "valid_roc", "test_roc", "train_pr", "valid_pr", "test_pr"]

        with open(log_path + "raw_results.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)

        results = np.array(results)
        results = [np.mean(results, axis=0), np.std(results, axis=0)]
        with open(log_path + "results.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)

        # Update cross validation log
        self.log["{}_B{}_C{}_D{}_L{}".format(model, batch, units_conv, units_dense, num_layers)] = results

        print(self)
        print("Training Ended")
