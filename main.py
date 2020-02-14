from Pipeline.pipeline import Pipeline

print("Start AutoML")

from pandas import read_csv

data = read_csv("Datasets/titanic.csv")
x = data.dtypes
pipeline = Pipeline()
result = pipeline.fit(data)
print(result)