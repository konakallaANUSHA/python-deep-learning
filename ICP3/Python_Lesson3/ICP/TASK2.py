import requests
from bs4 import BeautifulSoup

#getting the web content
html_doc = requests.get('https://beautiful-soup-4.readthedocs.io/en/latest/')

#scrapping content using BeautifulSoup
soup = BeautifulSoup(html_doc.text, 'html.parser')

print(soup.h1)
title = soup.title.string
print(title)
links = []
for link in soup.find_all('a'):
    links.append(link.get('href'))
    print(link)

#write in a file
with open('webscraping.txt', 'w') as f:
    f.write("Title: " + title + "\n")
    f.write("Links: " + "\n")
    for link in links:
        f.write("Link: " + str(link) + "\n")

print("webscraping txt is created and data is stored")
