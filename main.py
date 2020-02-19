from Pipeline.pipeline import Pipeline

print("Start AutoML")

from pandas import read_csv

# data = read_csv("Datasets/titanic.csv")
#
# pipeline = Pipeline()
# pipeline.fit(data)

# pipeline.save("pipeline1.json")



pipeline = Pipeline.load_pipeline("pipeline1.json")



data = read_csv("Datasets/titanic_test.csv")
result = pipeline.convert(data)
result.to_csv("Datasets/titanic_conv.csv", index=False)

