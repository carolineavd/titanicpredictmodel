import tflearn
import numpy

from tflearn.datasets import titanic

titanic.download_dataset("titanic_dataset.csv")

from tflearn.data_utils import load_csv
data, labels = load_csv("titanic_dataset.csv", target_column=0, categorical_labels=True, n_classes=2, columns_to_ignore=[2, 7])# two target columns/labels :survived or dead

for p in data:
        if p[1] == "female":
            p[1]=1
        else:
            p[1]=0

net = tflearn.input_data(shape=[None, 6])  # first layer of network has 6 layers
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)


# define model
model = tflearn.DNN(net)

model.fit(data, labels, n_epoch=5, batch_size=16, show_metric=True)
# data: passenger class, gender, age, siblings, parents/children, how much ticket cost
# model created to predict things
print(data[0][2])
# print("nate", model.predict([[2, 0, 40, 0, 0, 80.0]]))
print("caroline", model.predict([[1, 1, 40, 1, 2, 0.0]]))
# print("Caroline", model.predict)([[2, 1, 14, 1, 2, 0.0]])
