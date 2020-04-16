# Weights and Biases Sweep Timeout Example

This example shows how to run models and log the results to Weights and Biases while allowing your models to timeout after a certain amount of time.

![Demo Results](https://github.com/kevroy314/wandb-sweeps-timeout/raw/master/docs/demo.png)

## Setup

To setup, create an [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/) environment:

`conda create -n wandb_sweeps_timeout python=3.7 --yes`

activate it:

`activate wandb_sweeps_timeout`

install the required packages:

`pip install -r requirements.txt`

## Usage

To run, start the sweeps:

`wandb sweep sweep.yaml`

From there, you can follow the instructions in the [Weights and Biases documentation](https://docs.wandb.com/sweeps/quickstart).

## Key Issues

The `asyncmodels.py` file contains the asynchronous code. It's critical to note that the Weights and Biases summary logs will only populate from the main thread, not from the Process defined thread. As a result, retuning the critical summary methods to the main thread is required. You can still log to the history via the typical logging.
