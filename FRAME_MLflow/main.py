import mlflow

experiment_id = mlflow.set_experiment("MLflow Test2")
# experiment = mlflow.get_experiment_by_name("MLflow Test")

with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
        mlflow.log_param("child", "yes")
