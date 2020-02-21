from Pipeline.pipeline import Pipeline

print("Start AutoML")

####################### Pipeline example 1: processing data

from pandas import read_csv
data = read_csv("Datasets/house_train.csv")                     #read the raw data

pipeline = Pipeline()                                           #init a pipeline
result = pipeline.process(data)                                 #process the data (the mappings will be stored in the pipeline object)

pipeline.save("pipeline_house.json")                            #save the pipeline data
result.to_csv("Datasets/house_train_processed.csv", index=False)#write the processed dataframe to the csv

####################### Pipeline example 2: converting data

pipeline = Pipeline.load_pipeline("pipeline_house.json")        #load a pipeline previously daved
data = read_csv("Datasets/house_test.csv")                      #read the dataframe that needs to be converted
result = pipeline.convert(data)                                 #convert the data
result.to_csv("Datasets/house_test_converted.csv", index=False) #save the conversion to file

