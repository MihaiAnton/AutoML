from pandas import DataFrame

from Pipeline.DataProcessor.FeatureMapping.mapper import Mapper


class Engineer:
    """
        Class responsible with feature engineering.
        Handles numeric and categorical features ( textual to be added )

    """
    def __init__(self, config={}):
        self._config = config
        self._mapper = Mapper("Engineer")
        self._numeric_dtypes = ["float64", "int64", "bool", "float32", "int32", "int8", "float8"]
        self._textual_dtypes = ["object", "string"]

    def process(self, data:DataFrame, mapper: 'Mapper', column_type = {}):
        """
            Processes the dataset in a way that a learning algorithms can benefit more from it.
            Does outlier detection, feature engineering, data normalization/standardization, missing value filling, polynomial features and more.
        :param data: DataFrame consisting of the data
        :param mapper: parent mapper that keeps track of changes
        :param column_type: describes whether features are continuous or discrete in form of a dictionary
                            (if not provided, the algorithm will try to figure out by itself - may reduce overall performance)
        :return: processed data in form of DataFrame
        """

        #iterate through each column and process it according to it's type
        modified_data = DataFrame()
        col_types = data.dtypes

        for column in data.columns:
            dtype = col_types[column]

            interm_data = None
            if column in self._config.get("DO_NOT_PROCESS", []):
                interm_data = data[[column]]
            else:
                if dtype in self._numeric_dtypes:
                    #TODO process numeric column
                    pass
                elif dtype in self._textual_dtypes:
                    #TODO process textual column
                    pass
                else:
                    #TODO raise not known data type exception
                    pass

            if not (interm_data is None):
                modified_data.join(interm_data)                 #add the modified data to the new dataframe


        mapper.set_mapper(self._mapper)
        return modified_data


    def convert(self, data:DataFrame, mapper: 'Mapper'):
        """
            Converts new data to a format previously mapped into 'mapper'
        :param data: DataFrame with data to be transformed
        :param mapper: Mapper class containing the rules for transformation.
        :return: converted data in form of DataFrame
        """
