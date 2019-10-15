string = input("enter the string ")
current_string = ""
longest_string = ""
# initilize a set
s = set()
#loop through every character in string
for i in range(0, len(string)):
#check if the character is already in set
    c = string[i]
#if it exists then empty the longest_till_now and set
    if c in s:
        current_string = ""
        s.clear()
#if it doesnot exist then add to longest_till_now and to the set
    current_string = current_string + c
    s.add(c)
#check if the current substring lenght is greater than previous substring
#if it is then modify the overalllongest_substring
    if len(current_string) > len(longest_string):
        longest_string = current_string
print("the longest substring is", longest_string)
print("length is ", len(longest_string))