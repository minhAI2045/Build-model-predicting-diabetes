import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn



data = pd.read_csv("pima.csv")
# Function to check whether a value is a number or not
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Go through all cells in the DataFrame and delete cells containing characters
for col in data.columns:
    for index, value in data[col].items():
        if not is_number(value):
            data.at[index, col] = None

# Save the deleted DataFrame to a new Excel file
output_file_path = 'path_to_output_excel_file.xlsx'
data.to_excel(output_file_path, index=False)


data1= pd.read_csv("path_to_output_excel_file.csv")
target = "Class"
x = data1.drop(target, axis=1)
y = data1[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)


num_transformer = SimpleImputer(strategy="median")
result1 = num_transformer.fit_transform(x_train)
result2 = num_transformer.transform(x_test)


scaler = StandardScaler()
x_train = scaler.fit_transform(result1)
x_test = scaler.transform(result2)


cls1 = RandomForestClassifier()
cls1.fit(x_train,y_train)
y_predict1 = cls1.predict(x_test)
print(classification_report(y_test, y_predict1))
cm = np.array(confusion_matrix(y_test, y_predict1, labels=[1,0]))
confusion = pd.DataFrame(cm, index=[ "Diabetic","Not Diabetic"], columns=["Diabetic","Not Diabetic"])
sn.heatmap(confusion, annot=True)
plt.title("RandomForestClassifier")
plt.savefig("RandomForestClassifier")
plt.show()



cls2 = SVC()
cls2.fit(x_train,y_train)
y_predict2 = cls2.predict(x_test)
print(classification_report(y_test, y_predict2))
cm = np.array(confusion_matrix(y_test, y_predict2, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=["Diabetic", "Not Diabetic"], columns=["Diabetic", "Not Diabetic"])
sn.heatmap(confusion, annot=True)
plt.title("SVC")
plt.savefig("SVC")
plt.show()



cls3 = KNeighborsClassifier()
cls3.fit(x_train,y_train)
y_predict3 = cls3.predict(x_test)
print(classification_report(y_test, y_predict3))
cm = np.array(confusion_matrix(y_test, y_predict3, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=["Diabetic", "Not Diabetic"], columns=["Diabetic", "Not Diabetic"])
sn.heatmap(confusion, annot=True)
plt.title("KNeighborsClassifier")
plt.savefig("KNeighborsClassifier")
plt.show()



cls4 = DecisionTreeClassifier()
cls4.fit(x_train,y_train)
y_predict4 = cls4.predict(x_test)
print(classification_report(y_test, y_predict4))
cm = np.array(confusion_matrix(y_test, y_predict4, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=["Diabetic", "Not Diabetic"], columns=["Diabetic", "Not Diabetic"])
sn.heatmap(confusion, annot=True)
plt.title("DecisionTreeClassifier")
plt.savefig("DecisionTreeClassifier")
plt.show()
