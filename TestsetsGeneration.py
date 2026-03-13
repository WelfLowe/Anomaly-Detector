from testset_generator.DatasetGeneratorBlackout import DatasetGeneratorBlackout
from testset_generator.DatasetGeneratorLift import DatasetGeneratorLift
from testset_generator.DatasetGeneratorNoise import DatasetGeneratorNoise
from testset_generator.DatasetGeneratorSawtooth import DatasetGeneratorSawtooth
from testset_generator.DatasetGeneratorShift import DatasetGeneratorShift
from testset_generator.DatasetGeneratorAmplitude import DatasetGeneratorAmplitude
from testset_generator.DatasetGeneratorSkip import DatasetGeneratorSkip
import numpy as np

K = 1000
N = 500
anomaly_ratio = 0.05

dg = []
dg.append(DatasetGeneratorLift())
dg.append(DatasetGeneratorNoise())
dg.append(DatasetGeneratorShift())
dg.append(DatasetGeneratorAmplitude())
#dg.append(DatasetGeneratorSkip())
#dg.append(DatasetGeneratorSawtooth())
dg.append(DatasetGeneratorBlackout())

sev = np.linspace(0, 4, 30)
n_sets = 50

for n in range(n_sets):
    for i in range(len(dg)):
        for s in sev:
            #K data series a N points with 20% outliers of severeness degree s
            dg[i].generateKN(K,
                             N,
                             anomaly_ratio,
                             severeness=s,
                             verbose=False,
                             name="testsets_new_new/train_" + str(i) + "_" +
                             str(int(s * 29 / 4)) + "_" + str(n))
            dg[i].generateKN(int(K / 10),
                             int(N),
                             0.5,
                             severeness=s,
                             verbose=False,
                             name="testsets_new_new/val_" + str(i) + "_" +
                             str(int(s * 29 / 4)) + "_" + str(n))
