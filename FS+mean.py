
import pandas as pd
import numpy as np
data = pd.read_csv('diabetes.csv')
print(data.head())

# data= data.drop(['Pregnancies'], axis=1)
# data= data.drop(['BloodPressure'], axis=1)
# data= data.drop(['SkinThickness'], axis=1)
# data= data.drop(['BMI'], axis=1)
# data= data.drop(['DiabetesPedigreeFunction'], axis=1)
# data= data.drop(['Age'], axis=1)

# print(data)
# data.to_csv('diabetes3.csv',index=False, header=True)
# 检查每列是否存在缺失值
# print(data.isin([0]).sum())

# # 中位数替换缺失值
# median_value = data['Glucose'].median()
# data['Glucose'] = data['Glucose'].replace(0, median_value)
# print(data)
# median_value1 = data['Insulin'].median()
# data['Insulin'] = data['Insulin'].replace(0, median_value1)
#
# print(data)
# data.to_csv('median_value.csv',index=False, header=True)


# # 均值替换缺失值
# mean_value = data['Glucose'].mean()
# data['Glucose'] = data['Glucose'].replace(0, mean_value)
# print(data)
# mean_value1 = data['Insulin'].mean()
# data['Insulin'] = data['Insulin'].replace(0, mean_value1)
#
# print(data)
# data.to_csv('mean_value.csv',index=False, header=True)
# # 检查每列是否存在缺失值
# print(data.isin([0]).sum())

#检查每列是否存在缺失值
print(data.isin([0]).sum())

# 中位数替换缺失值
median_value = data['SkinThickness'].median()
data['SkinThickness'] = data['SkinThickness'].replace(0, median_value)
print(data)
data.to_csv('diabetes5.csv',index=False, header=True)

# 检查每列是否存在缺失值
print(data.isin([0]).sum())