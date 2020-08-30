import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(50):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

"""
for i in train_samples:
    print(i)

    49
    94
    31
    83
    13
    ...

for i in train_labels:
    print(i)

    0
    1
    0
    1
    0
    ...
"""

# convert to numpy array so can take it
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# the model may not learn very much from the data the way it is with numbers ranging from 13 to 100.
# We reshape the data as a technical formality just since the fit_transform() function doesn’t accept 1D data by default.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

"""
# The 49 moved to 0.47, 94 --> 0.609 etc...
for i in scaled_train_samples:
    print(i)
    
    [0.45977011]
    [0.81609195]
    [0.37931034]
    [0.90804598]
    [0.35632184]
    ...
"""
