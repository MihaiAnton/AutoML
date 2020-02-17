from Pipeline.pipeline import Pipeline

print("Start AutoML")

from pandas import read_csv

data = read_csv("Datasets/titanic.csv")

pipeline = Pipeline()
result = pipeline.fit(data)
pipeline.save_processor("processor.json")
print(result)