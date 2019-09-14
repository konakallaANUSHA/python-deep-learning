import numpy
from collections import Counter
frequent_list = []
array = numpy.random.randint(20, size=15)
print(array)
frequent = Counter(array)
print(frequent)

#to find the max key occuranace out of key,value
print("Most frequent items")
for key,value in frequent.items():
    if value == max(frequent.values()):
        print(key)

#replacing the maximum number in an array
replace = numpy.where(array == numpy.amax(array), 0, array)
print('Replacing Max No. by 0:', replace)

