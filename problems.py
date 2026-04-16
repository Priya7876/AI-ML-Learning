import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\priya\Downloads\archive\Titanic-Dataset.csv")
# Do males and females have different survival rates?

# df.groupby("Sex")["Survived"].mean().plot(
#     kind="bar" , color= ["#E85D24","#5DCAA5"]
# )

df[df["Survived"] ==1]["Age"].hist(label="Survived" , alpha = 0.5 , color = "#5DCAA5")
df[df["Survived"]==0]["Age"].hist(label= "Died" , alpha= 0.5 ,color = "#E85D24")

fig,axes = plt.subplots(2,2,figsize = (12,9))
plt.scatter(df["Fare"],df["Survived"] , alpha= 0.3 , cmap="RdYlGn" , c= df["Survived"])
plt.show()

