import pandas as pd 
import numpy as np

# salaries = pd.Series([10,20,40] , name = "Salary" )

# print(salaries)

# df = pd.DataFrame({
#     "Name" : ["Priya" , "Vansu" , "Saruchi" ,"Pallvi"],
#     "MathsScore" : [ 100,90,80 , np.nan],
#     "ScienceScore" : [80,60,np.nan ,100]
# })
# # print(df.shape)
# # print(df.dtypes)

# # print(df.head())
# # print(df.tail(1))
# # print(df.info())
# # print(df.describe())
# # print(df.isnull().sum())
# # print(df["Name"])
# # print(df[["Name" , "ScienceScore" , "ScienceScore"]])


# df[df["ScienceScore"] > 50]  ## So basically we can get the condtioonal values 
# print(df.loc[1,"Name"])

# print(df.isnull().sum())
# # Drop rows with nulls 
# df_dropped = df.dropna()
# # Fill the mean witj mean value 
# df["ScienceScore"] = df["ScienceScore"].fillna(df["ScienceScore"].mean())

# 

df = pd.DataFrame({
    "name":       ["Priya", "Rahul", "Sara", "Arjun", "Neha"],
    "age":        [28, np.nan, 35, 29, 32],
    "salary":     [85000, 92000, np.nan, 78000, 95000],
    "department": ["Engineering","Marketing","Engineering","Marketing","Engineering"],
    "joined":     ["2021-03-15","2019-07-01","2020-11-20","2022-01-10","2018-06-30"],
})

dept_stats = df.groupby("department")["salary"].agg(["mean", "count" , "max"])
print(dept_stats)

df["joined"] = pd.to_datetime(df["joined"])
