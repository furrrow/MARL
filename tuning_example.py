import optuna
from cleanrl_utils.tuner import Tuner
""" tuning_example.py from cleanrl
please see:
https://github.com/vwxyzjn/cleanrl/blob/master/tuner_example.py

notes:
put functions directly into 'if __name__ == '__main__':', do not use main()
- the parameter 'run_name' must be available from the top level of the function
Tuner's 'target_scores' needs to be populated
Suggest a couple trial runs with low 'total-timesteps' to test before kicking off all the runs
"""
tuner = Tuner(
    script="pettingzoo_ippo_cleanrl.py",
    metric="charts/episodic_return_listener_0",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        # "CartPole-v1": [0, 500],
        # "Acrobot-v1": [-500, 0],
        "simple_speaker_listener": [-500, 0],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 5e-5, 3e-3, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [1, 2, 4]),
        "update-epochs": trial.suggest_categorical("update-epochs", [4, 8, 10, 12]),
        "num-steps": trial.suggest_categorical("num-steps", [128, 264, 1024, 2056]),
        "vf-coef": trial.suggest_float("vf-coef", 0, 5),
        "max-grad-norm": trial.suggest_float("max-grad-norm", 0, 5),
        "total-timesteps": 400_000,
        "num-envs": 1,
    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)