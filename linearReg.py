from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
salary     = np.array([40000, 45000, 50000, 55000, 62000, 70000,
                       76000, 83000, 90000, 98000, 105000, 112000,
                       118000, 125000, 132000])



experience = experience.reshape(-1,1)


x_test , x_train , y_test , y_train = train_test_split(
    experience , 
    salary,
    test_size=0.2,
    random_state = 42
)
model = LinearRegression()
model.fit(x_train , y_train )

predicted =model.predict(x_test)
print(f" Here is the predicted value {predicted} ")

print(f"Here is the actual value {y_test}")

error = mean_squared_error(y_test , predicted)
rmse = np.sqrt(error)
print(f"here is the erro{rmse}")
score = r2_score(y_test , predicted)
print(f"Here is the score {score}")


plt.figure(figsize=(8,4))
plt.scatter(experience , salary ,alpha=0.3 , color= "green" , label = " Actula data")
x_label = np.linspace(1,50,5).reshape(-1,1)
y_label = model.predict(x_label)
plt.plot(x_label , y_label , color = "red" )
plt.show()

sal = model.predict([8])
print(f"Here is the salary  of 8 workx {sal} ")
