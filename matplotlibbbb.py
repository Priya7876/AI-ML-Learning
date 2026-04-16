import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\priya\Downloads\archive\Titanic-Dataset.csv")
print(df.head())
print(df.isnull().sum())
df=df.drop(columns = ["Cabin"])
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df)
# print(df.isnull().sum())

# survivalRate = df["Survived"].mean()
# print(f"Here is the {survivalRate:.1%}")

# # Survival by gender
# survivalByGender = df.groupby("Sex")["Survived"].mean()
# print(f"Here is the survival rate by gender {survivalByGender}")

# # Survival by class
# survivalByClass = df.groupby("Pclass")["Survived"].agg(["mean","sum" , "count"]).round(2)
# print(f"Here is the survival rate by class{survivalByClass}")

# #  Avg Age 
# ageSurvive = df.groupby("Survived")["Age"].mean()
# print(f"Here is the ageSurvive {ageSurvive}")

# fig,axes = plt.subplots(2,2,figsize = (12,9))
# fig.suptitle("Titanic Survival Analysis" , fontsize=15 , fontweight= "bold")

# df.groupby("Sex")["Survived"].mean().plot(
# kind = "bar" , ax= axes[0,0] , color = ["#E85D24","#5DCAA5"] ,
# edgecolor= "white" , rot =0

# )
# axes[0,0].set_title("Survival rate by gender")

# df[df["Survived"]==1]["Age"].hist(ax=axes[1,0],
#                                   bins= 25, alpha=0.6,label= "Survived",
#                                   color = "#5DCAA5"
                                  
                                  
#                                   )
# df[df["Survived"]==0]["Age"].hist(ax=axes[1,0],
#                                   bins= 25, alpha=0.6,label= "Died",
#                                   color = "#E85D24"
                                  
                                  
#                                   )

# axes[1,0].set_title ("Age distribution by survival")




# plt.show()

