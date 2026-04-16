from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
import pandas as pd
import numpy as py 

data = {
    "text": [
        "Win money now",
        "Call me today",
        "Free prize waiting",
        "Let's go for lunch",
        "Earn cash fast",
        "Meeting at 5pm" , 
        "Hello ! whats up",
        "Do this and money","Can we meet",
        "Go and take your money"
    ],
    "label": [1, 0, 1, 0, 1, 0 , 0, 1,0,1]
}

df = pd.DataFrame(data)
vecInput = CountVectorizer()
x_label = vecInput.fit_transform(df["text"])
y_label = df["label"]

model = LogisticRegression()
x_train, x_test , y_train , y_test = train_test_split(x_label ,
                                                       y_label,
                                                       test_size=0.2,
                                                       random_state=42
                                                      )
model.fit(x_train, y_train)
predicted = model.predict(x_test)

print(f"here is the predicted value {predicted}")

print (f"Here is the actual value {y_test}")

loss = log_loss(y_test , predicted)

print(f"here is  the loss {loss}")

