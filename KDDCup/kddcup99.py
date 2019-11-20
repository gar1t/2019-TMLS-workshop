import argparse
import errno
import os

import pandas as pd

from sklearn import preprocessing

import h2o

class TrainData:

    def __init__(self, x_cols, y_col, train, val):
        self.x_cols = x_cols
        self.y_col = y_col
        self.train = train
        self.val = val

def main():
    args = _init_args()
    _init_h2o(args)
    train_data = _init_train_data(args)
    model = _init_model(args)
    _train_model(model, train_data, args)
    _print_train_results(model)
    _save_model(model, args)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="kddcup99.csv")
    p.add_argument(
        "--val-split", type=float, default=0.2,
        help="percent data used for validation")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output", default="output")
    p.add_argument("--model-name", default="model.h2o")
    p.add_argument("--h2o-mem", default="25g")
    return p.parse_args()

def _init_h2o(args):
    h2o.init(min_mem_size=args.h2o_mem)

def _init_train_data(args):
    raw_data = _load_raw_data(args)
    x_cols, y_col = _prepare_raw_data(raw_data)
    return _train_data(x_cols, y_col, raw_data, args)

def _load_raw_data(args):
    return pd.read_csv(args.data)

def _prepare_raw_data(raw_data):
    y_col = "label"
    x_cols = raw_data.drop(y_col, axis=1).columns.tolist()
    raw_data[y_col] = raw_data[y_col].apply(
        lambda x: x if x == "normal" else "zAttack")
    raw_data[y_col] = _fit_transform(raw_data[y_col])
    return x_cols, y_col

def _fit_transform(col):
    return preprocessing.LabelEncoder().fit_transform(col)

def _train_data(x_cols, y_col, raw_data, args):
    train_data = h2o.H2OFrame(raw_data)
    for name in raw_data.columns.tolist():
        if raw_data[name].dtypes == "object":
            train_data[name] = train_data[name].asfactor()
    train_data[y_col] = train_data[y_col].asfactor()
    train_split = 1 - args.val_split
    train, val = train_data.split_frame([train_split], seed=args.random_seed)
    return TrainData(x_cols, y_col, train, val)

def _init_model(args):
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    return H2OGeneralizedLinearEstimator(
        nfolds=args.n_folds,
        family="binomial",
        lambda_search=False,
        seed=args.random_seed)

def _train_model(model, data, _args):
    model.train(
        data.x_cols,
        data.y_col,
        training_frame=data.train,
        validation_frame=data.val)

def _print_train_results(model):
    print("AUC: %f" % model.auc(valid=True))

def _save_model(model, args):
    _ensure_dir(args.output)
    tmp_path = h2o.save_model(model=model, path=args.output)
    model_path = os.path.join(args.output, args.model_name)
    os.rename(tmp_path, model_path)

def _ensure_dir(path):
    try:
        os.makedirs(path)
    except IOError as e:
        if e.errno != errno.EEXIST:
            raise

if __name__ == "__main__":
    main()
