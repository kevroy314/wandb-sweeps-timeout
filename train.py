import time

from asyncmodels import AsyncModelTimeout

import wandb

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparameter_defaults = dict(
    max_depth=None,
    n_estimators=100
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
config = wandb.config


# Make some sample data
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=42, shuffle=False)
						   
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


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

print(success)
print(f1_score)
