import requests
from bs4 import BeautifulSoup
html = requests.get("https://beautiful-soup-4.readthedocs.io/en/latest/")
bsObj = BeautifulSoup(html.content, "html.parser")
a= bsObj.find(id="making-the-soup" )
print(a)
print(bsObj.h1)
title = bsObj.title.string
print(title)

'''print(bsObj.get_text())'''
links = []
for link in bsObj.find_all('a'):
 print(link.get('href'))