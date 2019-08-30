# creating an empty list
lst = []
opy = []

# number of elemetns as input
n = int(input("Enter number of elements : "))

# iterating till the range
for i in range(0, n):
    ele = int(input("Enter weight in lbs: "))

    lst.append(ele)  # adding the element

print(lst)

for i in lst:
    y = 0.4538 * i

    print(y)

    opy.append(y)

print(opy)