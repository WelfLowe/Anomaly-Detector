import numpy as np
from scipy.stats import differential_entropy

DS = '3_29_25'

data = np.load(f'testsets_new/val_{DS}.npy')
labels = np.load(f'testsets_new/val_{DS}_labels.npy')

data0 = data[labels == 0]
data1 = data[labels == 1]
print(data0.shape)
print(data1.shape)

ent0 = []
ent1 = []
for _ in range(95):
    ent0.append(differential_entropy(data0[_]))

for _ in range(5):
    ent1.append(differential_entropy(data1[_]))

print(f'Label 0: {np.mean(ent0)}, Label 1: {np.mean(ent1)}')
