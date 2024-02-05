import pandas as pd

data = pd.read_csv("pima.csv")    # #load the dataset
data.head() #show the first 5 rows from the dataset

#checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed
data.info()