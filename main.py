import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    # if isinstance(config["main"]["execute_steps"], str):
    #     # This was passed on the command line as a comma-separated list of steps
    #     steps_to_execute = config["main"]["execute_steps"].split(",")
    # else:
    #     assert isinstance(config["main"]["execute_steps"], list)
    #     steps_to_execute = config["main"]["execute_steps"]

    # this project included python 3.13 which is too new
    # ideally should use 3.10 at time of writing as MLFlow and Hydra are not fully stable together yet
    steps_to_execute = config["main"]["execute_steps"]

    if isinstance(steps_to_execute, str):
        steps_to_execute = steps_to_execute.split(",")
    else:
        steps_to_execute = list(steps_to_execute)

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        # call the preprocess step
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "preprocessed_data",
                "artifact_description": "Data with preprocessing applied"
            }
        )

    # this is where deterministic and non-deterministic tests are happening
    # non-deterministic tests need a reference artifact and a sample artifact to compare against
    # to determine what is the statistical difference between the datasets
    # see config.yaml "reference_dataset"
    if "check_data" in steps_to_execute:

        # call the check_data step
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                # output of the previous "preprocess" step
                "sample_artifact": "preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"],
            }
        )

    # train/test split
    if "segregate" in steps_to_execute:

        # call the segregate step
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact": "preprocessed_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                # see config.yaml
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"],
            }
        )

    # model train and validation and export model step
    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        # call the random_forest step
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"],
            }
        )

    # take the trained model and test it against the test dataset
    if "evaluate" in steps_to_execute:

        # call the evaluate step
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest",
            }
        )


if __name__ == "__main__":
    go()
