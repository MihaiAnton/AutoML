from pandas import read_csv
from Pipeline import Pipeline, load_pipeline, load_model

print("Start AutoML")

####################### Pipeline example 1: processing data


# data = read_csv("Datasets/titanic.csv")  # read the raw data
#
# pipeline = Pipeline()  # init a pipeline
# result = pipeline.process(data)  # process the data (the mappings will be stored in the pipeline object)
#
# pipeline.save("pipeline_titanic.bin")  # save the pipeline data
# result.to_csv("Datasets/titanic_proc.csv", index=False)  # write the processed data frame to the csv

####################### Pipeline example 2: converting data

# pipeline = load_pipeline("pipeline_titanic.bin")  # load a pipeline previously saved
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
# model.train(X, Y, 10)
# pred = model.predict(X)
# print(1)
####################### Pipeline example 4: train a default model using the pipeline


data = read_csv("Datasets/titanic_converted.csv")

pipeline = Pipeline(dynamic=True)  # create a pipeline
model = pipeline(data)            # learn from the data





model.save("model")
model = load_model("model")  # reload the model
data = data.drop("Survived", axis=1)  # read the data and drop the predicted col

pred1 = model.predict(data)  # create a prediction
pipeline.save("pipeline.bin")
del pipeline

pipeline = load_pipeline("pipeline.bin")
model2 = pipeline.get_model()


pred2 = model2.predict(data)  # generate a second prediction

diff = (pred1 != pred2).any()[0]  # the 2 predictions should be identical

print("Works as expected!" if not diff else "Differences in predictions!")
