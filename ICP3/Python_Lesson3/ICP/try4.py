import numpy
a = numpy.random.randint(20, size=20)

print(a)

array = a.reshape((2,10))

print(array)

maxInRows = numpy.amax(array, axis=1)
print('Max value of every Row: ', maxInRows)

replace = numpy.where(array == numpy.amax(array, axis=1, keepdims=True), 0, array)

print('Replacing Max No. by 0:', replace)
