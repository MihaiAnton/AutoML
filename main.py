from Pipeline.pipeline import Pipeline

print("Start AutoML")

from pandas import read_csv

data = read_csv("Datasets/titanic.csv")

# pipeline = Pipeline()
# pipeline.fit(data)

data = data.drop(["Survived"],axis=1)
# result = pipeline.convert(data)
# pipeline.save_processor("processor.json")
# result.to_csv("Datasets/titanic_conv.csv")

from Pipeline.DataProcessor.processor import Processor
p = Processor.load_processor("processor.json")
result = p.convert(data)
result.to_csv("Datasets/titanic_conv.csv")

# from Pipeline.DataProcessor.processor import Processor
# proc = Processor.load_processor("processor.json")
# proc.save_processor("processor2.json")
