from testset_generator.DatasetGeneratorBlackout import DatasetGeneratorBlackout
from testset_generator.DatasetGeneratorLift import DatasetGeneratorLift
from testset_generator.DatasetGeneratorNoise import DatasetGeneratorNoise
from testset_generator.DatasetGeneratorSawtooth import DatasetGeneratorSawtooth
from testset_generator.DatasetGeneratorShift import DatasetGeneratorShift
from testset_generator.DatasetGeneratorAmplitude import DatasetGeneratorAmplitude
from testset_generator.DatasetGeneratorSkip import DatasetGeneratorSkip

K = 10
N = 500

#K data series a N points with 20% outliers of severeness degree 2
dg = DatasetGeneratorLift()
anomaly_ratio = 0.05
#testCase0 = dg.generateKN(K, N, anomaly_ratio, severeness=2, verbose = False, name ="testCase0")

dg = DatasetGeneratorNoise()
testCase1 = dg.generateKN(K, N, 0.2, severeness=2, verbose = False)

dg = DatasetGeneratorShift()
testCase2 = dg.generateKN(K, N, 0.2, severeness=2, verbose = False)

dg = DatasetGeneratorAmplitude()
testCase3 = dg.generateKN(K, N, 0.2, severeness=2, verbose = False)

dg = DatasetGeneratorSkip()
testCase4 = dg.generateKN(K, N, 0.2, severeness=2, verbose = False)

dg = DatasetGeneratorSawtooth()
testCase5 = dg.generateKN(K, N, 0.2, severeness=2, verbose = False)

dg = DatasetGeneratorBlackout()
testCase6 = dg.generateKN(K, N, 0.2, severeness=2, verbose = True)