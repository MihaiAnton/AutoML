from Pipeline.pipeline import Pipeline

print("Start AutoML")

####################### Pipeline example 1: processing data

from pandas import read_csv

# data = read_csv("Datasets/titanic.csv")  # read the raw data
#
# pipeline = Pipeline()  # init a pipeline
# result = pipeline.process(data)  # process the data (the mappings will be stored in the pipeline object)
#
# pipeline.save("pipeline_titanic.json")  # save the pipeline data
# result.to_csv("Datasets/titanic_proc.csv", index=False)  # write the processed data frame to the csv
#
# ####################### Pipeline example 2: converting data
#
# pipeline = Pipeline.load_pipeline("pipeline_titanic.json")  # load a pipeline previously saved
# data = read_csv("Datasets/titanic.csv")  # read the data frame that needs to be converted
# result = pipeline.convert(data)  # convert the data
# result.to_csv("Datasets/titanic_converted.csv", index=False)  # save the conversion to file


####################### Pipeline example 3: train default model by using the model class
# from Pipeline.Learner.Models.SpecializedModels.deepLearningModel import DeepLearningModel
# from Pipeline.DataProcessor.DataSplitting.splitter import Splitter
#
# data = read_csv("Datasets/titanic_converted.csv")
# X, Y = Splitter.XYsplit(data, "Survived")
#
# model = DeepLearningModel(X.shape[1], 1)
# model.train(X, Y, 60)       #TODO not using time at this moment - make use of it

####################### Pipeline example 4: train a default model using the pipeline
pipeline = Pipeline()

data = read_csv("Datasets/titanic_converted.csv")
model = pipeline.learn(data)

data = data.drop("Survived", axis=1)
x = model.predict(data)
print(x)




























