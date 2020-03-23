# this script provides an example on how to use the pipeline with the evolutionary algorithms
# the code is similar to the code used for the other approaches, but all the changes rely in the configuration file

from pandas import read_csv
from Pipeline import Pipeline

data = read_csv("../Datasets/titanic.csv")

# create a pipeline with the default configuration
pipeline = Pipeline()

# fit the data to the pipeline
model = pipeline.fit(data)

# save the model for further reusage
model_save_file = "./models/titanic_evol_model"
model.save(model_save_file)

# the pipeline can also be saved for further use in data conversion, training and prediction
# beware though that when using the saved pipeline for training, a new training session will
#   begin, despite the model that was saved with the pipeline
#   the model within the pipeline is used only to make predictions
#   if one wants to further train the model from the pipeline, it can do so by retrieving the model and
#       calling train() on it
pipeline_save_file = "./pipelines/titanic_evol_file"
pipeline.save(pipeline_save_file)
