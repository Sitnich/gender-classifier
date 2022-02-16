import os
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import src.models.train as tr

root_dir = os.path.abspath("..")


def ray_tuning():
    num_samples = 20
    max_num_epochs = 20
    gpus_per_trial = 0

    if not os.path.isdir(root_dir + '/data/tuning'):
        os.mkdir(root_dir + '/data/tuning')

    data_dir = os.path.abspath(root_dir + "/data/tuning")

    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([4, 6, 8, 12, 16]),
        "epochs": tune.choice([10, 20])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "epochs"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(tr.train_n_tune, data_dir=data_dir),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
