import optuna
import joblib

study = optuna.load_study(study_name="vq_sarsa_tuning", storage="vq_sarsa_tuning_data/vq_sarsa_tuning_study.pkl")

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
