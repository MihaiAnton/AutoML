from Pipeline.pipeline import Pipeline

print("Start AutoML")

from pandas import read_csv
# data = read_csv("Datasets/Iris.csv")
#
# pipeline = Pipeline()
# result = pipeline.process(data)
#
# pipeline.save("pipeline_iris.json")
# result.to_csv("Datasets/Iris_processed.csv", index=False)



pipeline = Pipeline.load_pipeline("pipeline_iris.json")



data = read_csv("Datasets/Iris.csv")
result = pipeline.convert(data)
result.to_csv("Datasets/Iris_converted.csv", index=False)

