def my_function():



 ip = input("Please insert characters: ")
 even_string=''
 odd_string=''
 for index in range(len(ip)):
    if index % 2 != 0:
        odd_string = odd_string+ip[index]
    else:
        even_string = even_string+(ip[index])
 print(even_string)
 print(odd_string)



my_function()