import sys
import getopt
import torch
import deepmatcher as dm
import deepmatcher.optim as optim
from model.HierMatcher import *
import os
gpu_no = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


def run_experiment(model_name, dataset_dir, embedding_dir):
    train_file = "train.csv"
    valid_file = "valid.csv"
    test_file = "test.csv"
    datasets = dm.data.process(path=dataset_dir,
                              train=train_file,
                              validation=valid_file,
                              test=test_file,
                              embeddings_cache_path=embedding_dir)

    train, validation, test = datasets[0], datasets[1], datasets[2] if len(datasets)>=3 else None

    if model_name == "HierMatcher":
        model = HierMatcher(hidden_size=150,
                            embedding_length=300,
                            manualSeed=2)

    model.run_train(train,
                    validation,
                    epochs=15,
                    batch_size=64,
                    label_smoothing=0.05,
                    pos_weight=1.5,
                    best_save_path='best_model_.pth' + gpu_no + '.pth')

    if test is not None:
        model.run_eval(test)


def get_params(argv):
    model_name = ""
    dataset_dir = ""
    embedding_dir = ""

    try:
        opts, args = getopt.getopt(argv, "hm:d:e:", ["help","model_name", "dataset_dir", "embedding_dir"])
    except getopt.GetoptError:
        print('python run.py -m <model_name> -d <dataset_dir> -e <embedding_dir> ')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('python run.py -m <model_name> -d <dataset_dir> -e <embedding_dir> ')
            sys.exit()
        if opt in ("-m", "--model_name"):
            model_name = arg
            print("model_name:", model_name)
        if opt in ("-d", "--dataset_dir"):
            dataset_dir = arg
            print("dataset_dir:", dataset_dir)
        if opt in ("-e", "--embedding_dir"):
            embedding_dir = arg
            print("embedding_dir:", embedding_dir)
    return model_name, dataset_dir, embedding_dir


if __name__ == '__main__':
    model_name, dataset_dir, embedding_dir = get_params(sys.argv[1:])
    if model_name != "" and dataset_dir != "" and embedding_dir != "":
        run_experiment(model_name, dataset_dir, embedding_dir)


