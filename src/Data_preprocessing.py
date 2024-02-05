import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



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