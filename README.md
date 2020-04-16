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

## Critical Code

The most critical bit of this example is the wrapping of your training code in the `AsyncModelTimeout` object which creates a separate process which can time out. Your normal training code should be wrapped in a function. You can then wrap that in a lambda or use globals to get the `config` and data to your model. That function can then be used with the `AsyncModelTimeout` object which will return the results of your function and a `success` flag (True if the model did not time out).

```

# Define how to run the model asynchronously
def run_model(config, X_train, X_test, y_train, y_test):
    # Just a demo model, but it needs the config
    clf = RandomForestClassifier(random_state=42, **config)
    
    # Time the fit predictions
    t0 = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fit_predict_time = time.time() - t0
    
    # Score it
    score = f1_score(y_test, y_pred)
    wandb.log({"fit_predict_time": fit_predict_time}) # Logged to wandb history
    
    return score # This will be returned out to the wandb summary

run_model_with_parameters = lambda: run_model(config, X_train, X_test, y_train, y_test)

async_model = AsyncModelTimeout(run_model_with_parameters, 30) # 30 second runtime limit per job
success, score = async_model.run()
wandb.log({"model_completed": success}) # False if the model had to stop early
wandb.log({"f1_score": score})
```

## Key Issues

The `asyncmodels.py` file contains the asynchronous code. It's critical to note that the Weights and Biases summary logs will only populate from the main thread, not from the Process defined thread. As a result, retuning the critical summary methods to the main thread is required. You can still log to the history via the typical logging.
