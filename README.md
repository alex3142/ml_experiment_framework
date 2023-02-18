# ML Experiment Framework

This code is a lightweight framework to run and track experiments and deploy models via an API

## Set Up

This code assumes [anaconda](https://www.anaconda.com/) is available on
the host machine. If not it is available for download at the link above.
The below instructions assume a Windows 11 operating system. 
The below commands should be run inside a 

#### Create Environment
`conda create -n name-of-env python=3.10`

#### Activate Environment
`conda activate name-of-env`

#### Install Dependencies
`pip install -r name_of_requirements.txt`

#### Download Spacy English Language Model (if doing NLP)
`python -m spacy download en_core_web_sm`

#### Create Jupyter Kernel
`python -m ipykernel install --user --name name-of-env`


## Folder Structure Files

Contained in the repo are the following folders:

- data
  - This folder contains the raw data, and the processed data
- data_processing 
  - This folder contains the processor functionality for processing the raw text
- data_review_and_processing
  - This folder contains the code the review and split that data, 
  then review the data, and since the data needed to be processed to perform the required analysis
  this processing is also done in the eda script.
- deployment 
  - This folder contains the code to deploy the model as an API endpoint
- modelling
  - contains the code needed to run experiments to train and assess models
  - experiment configs
    - this folder contains the configuration files used to run experiments
  - mlruns
    - This folder contains the results of the experiments as captured by mlflow
  - model_bases
    - This folder contains the base models used in modelling
  - models
    - This folder contains the serialised models saved from experimentation
  - tests
    - This folder contains the tests for the above code



## How to Run Code

Set up the environment as described above.

### Run Experiments 

Then the experiment code is designed to run as a command line tool. The repo will need to be 
pip installed into an environment, then from the command line cd into the `modelling` folder
and run `python run_experiment.py --config_path name_of_experiment_config_file.yaml`.

### View Experiment Results 

Then the results of the experiment can be found by running `mlflow ui` in the terminal
to run the tracking server locally (note, by default this runs on port 5000).

### Deploying Models

To deploy a model the full path to the model pickle file should be added as an environment 
variable called `model_path`, in the `model_envs_example` file 
by default this will run the api at localhost port 5016.
In order to change the ip or port `host` and `port` can be added as environment variables.

The git repo should be added to the docker file in the deployment folder
then inside the folder cd to the deployment folder and run:

`docker build -t container_name:container.version . `

`docker run --env-file=model_envs_file_name -p 5016:5016 container_name:container.version`


Finally to query the API send a post request to the designated `host:port/predict` with 'review_text' mapping
to a string of text to be processed and the response will contain the key 'prediction' with the
review prediction. 