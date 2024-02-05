import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

data1= pd.read_csv("path_to_output_excel_file.csv")
data1.hist()
data1.plot(kind='density',subplots= True, layout=(3,3),sharex= False)
sn.heatmap(data1.corr(), annot= True)
plt.show()