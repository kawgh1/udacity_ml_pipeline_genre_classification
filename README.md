# Setting up the venv to run the mlflow pipeline

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate.bat    # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install mlflow scikit-learn hydra-core omegaconf wandb

# Log in to W&B
wandb login

# Run the MLflow pipeline locally
mlflow run . --env-manager=local

```

```bash
wandb: Syncing run frosty-night-20
wandb: ⭐️ View project at https://wandb.ai/krseven-j/exercise_14
wandb: 🚀 View run at https://wandb.ai/krseven-j/exercise_14/runs/clzje4kk
2026-02-12 07:39:54,035 Creating artifact
2026-02-12 07:39:54,066 Logging artifact
...
...
2026-02-12 07:39:58,600 Downloading artifact
2026-02-12 07:39:59,674 Dropping duplicates
2026-02-12 07:39:59,698 Feature engineering
...
...
cachedir: .pytest_cache
rootdir: /Users/.../Desktop/udacity_ml_pipeline_genre_classification/check_data
collected 4 items                                                                                                                                       

test_data.py::test_column_presence_and_type PASSED
test_data.py::test_class_names PASSED
test_data.py::test_column_ranges PASSED
test_data.py::test_kolmogorov_smirnov PASSED
========== 4 passed, 6 warnings in 2.56s ======
wandb: 
wandb: 🚀 View run polished-brook-22 at: https://wandb.ai/krseven-j/exercise_14/runs/x4f089y9
...
...
wandb: Syncing run sleek-donkey-23
wandb: ⭐️ View project at https://wandb.ai/krseven-j/exercise_14
wandb: 🚀 View run at https://wandb.ai/krseven-j/exercise_14/runs/6bvkhdzo
2026-02-12 07:40:12,997 Downloading and reading artifact
2026-02-12 07:40:13,897 Splitting data into train, val and test
2026-02-12 07:40:13,925 Uploading the train dataset to data_train.csv
2026-02-12 07:40:14,116 Logging artifact
2026-02-12 07:40:15,488 Uploading the test dataset to data_test.csv
2026-02-12 07:40:15,591 Logging artifact
...
...
wandb: Syncing run fast-sea-24
wandb: ⭐️ View project at https://wandb.ai/krseven-j/exercise_14
wandb: 🚀 View run at https://wandb.ai/krseven-j/exercise_14/runs/qr8exn5u
2026-02-12 07:40:22,454 Downloading and reading train artifact
2026-02-12 07:40:23,405 Extracting target from dataframe
2026-02-12 07:40:23,407 Splitting train/val
2026-02-12 07:40:23,427 Setting up pipeline
2026-02-12 07:40:23,430 Fitting
2026-02-12 07:40:33,753 Scoring
...
...
wandb: Syncing run serene-brook-25
wandb: ⭐️ View project at https://wandb.ai/krseven-j/exercise_14
wandb: 🚀 View run at https://wandb.ai/krseven-j/exercise_14/runs/488hy8wh
2026-02-12 07:40:45,904 Downloading and reading test artifact
2026-02-12 07:40:46,729 Extracting target from dataframe
2026-02-12 07:40:46,730 Downloading and reading the exported model
wandb:   7 of 7 files downloaded.  
2026-02-12 07:40:48,122 Scoring
2026-02-12 07:40:48,205 Computing confusion matrix
...
...
2026/02/12 07:40:49 INFO mlflow.projects: === Run (ID 'dabf439175a74693b271fcd7f764c7d9') succeeded ===
2026/02/12 07:40:50 INFO mlflow.projects: === Run (ID '61719c7c6e774ede8a45b491f5ef8717') succeeded ===

```

![screenshot.png](screenshot.png)

Then run it for `prod`

```bash
mlflow run . --env-manager=local
-P hydra_options="main.project_name=genre_classification_prod"
```

```bash
...
...
2026/02/12 07:52:55 INFO mlflow.projects: === Run (ID 'b7160d098cc2475c9d6c5a8f0900d8be') succeeded ===
2026/02/12 07:52:55 INFO mlflow.projects: === Run (ID '867476d70f85448681fbabb08e57e816') succeeded ===
...
...
```


![pipeline-deployment-graph.png](pipeline-deployment-graph.png)

<br>
<br>
<br>

## Run this ML pipeline from a GitHub release
After install the required dependencies above and or creating a new venv
Create a new release based on this branch in GitHub
> mlflow run  -v [version] [github URL] -P ...

Run:
```bash
mlflow run -v 1.0.2  git@github.com:kawgh1/udacity_ml_pipeline_genre_classification.git
-P hydra_options="main.project_name=remote_execution"   
```

where `1.0.2` is the release tag. We also change the project name so it does not overwrite any existing projects in `wandb`.

<br>

```
2026-02-14 14:21:21,463 Downloading and reading test artifact
2026-02-14 14:21:22,461 Extracting target from dataframe
2026-02-14 14:21:22,462 Downloading and reading the exported model
wandb:   7 of 7 files downloaded.  
2026-02-14 14:21:24,339 Scoring
2026-02-14 14:21:24,426 Computing confusion matrix
wandb: 
wandb: 🚀 View run sparkling-darling-16 at: https://wandb.ai/krseven-j/remote_execution/runs/l1xghvoh
wandb: Find logs at: wandb/run-20260214_142120-l1xghvoh/logs
2026/02/14 14:21:26 INFO mlflow.projects: === Run (ID 'd9a2d20fdb7c4f3c948b42ac629a3054') succeeded ===
2026/02/14 14:21:26 INFO mlflow.projects: === Run (ID '0618429d31954bf4aa3f74e111280112') succeeded ===
```

[https://wandb.ai/krseven-j/remote_execution/runs/l1xghvoh](https://wandb.ai/krseven-j/remote_execution/runs/l1xghvoh)

<br>
<br>

## Other Notes
After running the pipeline successfully, go to W&B and tag the exported model as ``prod`` as
we did in Exercise 13.

A few notes and instructions:
* When chaining together the steps, the output artifact of a step should be the input artifact
  of the next one (when applicable). Also use the ``artifact_type`` options so that the final
  visualization of the pipeline highlights the different steps. For example, you can use
  ``raw_data`` for the artifact containing the downloaded data, ``preprocessed_data`` for the
  artifact containing the data after the preprocessing, and so on.
  
* For testing, set the ``project_name`` to ``exercise_14``. Once you are done
  developing, do a production run by changing the ``project_name`` to 
  ``genre_classification_prod``. This way the visualization of the pipeline will not contain 
  all your trials and errors. Remember to tag the produced model export as ``prod`` (we are going
  to use it in the next exercise)
  
* When developing, you can override the parameter ``main.execute_steps`` to only execute one or
  more steps of the pipeline, instead of the entire pipeline. This is useful for debugging. 
  For example, this only executes the ``random_forest`` step:
  ```bash
  mlflow run . -P hydra_options="main.execute_steps='random_forest'"
  ```
  and this executes ``download`` and ``preprocess``:
  ```bash
  mlflow run . -P hydra_options="main.execute_steps='download,preprocess'"
  ```


