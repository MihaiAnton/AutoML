from Pipeline.pipeline import Pipeline

print("Start AutoML")

from pandas import read_csv
# data = read_csv("Datasets/house_train.csv")
#
# pipeline = Pipeline()
# result = pipeline.process(data)
#
# pipeline.save("pipeline_house.json")
# result.to_csv("Datasets/house_train_processed.csv", index=False)



# pipeline = Pipeline.load_pipeline("pipeline_house.json")
# data = read_csv("Datasets/house_test.csv")
# result = pipeline.convert(data)
# result.to_csv("Datasets/house_test_converted.csv", index=False)
# #

d1 = read_csv("Datasets/house_train_processed.csv")
d2 = read_csv("Datasets/house_test_converted.csv")
print(d1.shape, d2.shape)
print(d1.columns.tolist())
print(d2.columns.tolist())
