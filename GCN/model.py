from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from GCN.layer import *
from GCN.loss import *


def GCN(hyper):
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))

    out = atoms
    for _ in range(num_layers):
        out = GCNGraphConv(units_conv, activation='relu')([out, adjms])

    out = GCNGraphGather(pooling='sum')(out)
    out = Dense(units_dense, activation='relu')(out)
    out = Dense(units_dense, activation='relu')(out)

    if task == "regression":
        out = Dense(1, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, adjms], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(1, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, adjms], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(1, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, adjms], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model
