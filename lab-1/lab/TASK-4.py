from bs4 import BeautifulSoup
import requests
import csv

# Step 1: Sending a HTTP request to a URL
url = "https://scikit-learn.org/stable/modules/clustering.html#clustering"
# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Step 2: Parse the html content
soup = BeautifulSoup(html_content, "lxml")



# Get the table having the class wikitable
gdp_table = soup.find("table", attrs={'class':'colwidths-given'})
gdp_table_data = gdp_table.tbody.find_all("tr")  # contains 2 rows



t_headers = []
for th in gdp_table.find_all("th"):
        # remove any newlines and extra spaces from left and right
        t_headers.append(th.text.replace('\n', ' ').strip())
print('HEADERS')
print(t_headers)

# Get all the rows of table

table_data = []
for tr in gdp_table.tbody.find_all("tr"):  # find all tr's from table's tbody
    t_row = {}
        # Each table row is stored in the form of
        # t_row = {'Rank': '', 'Country/Territory': '', 'GDP(US$million)': ''}

        # find all td's(3) in tr and zip it with t_header
    for td, th in zip(tr.find_all("td"), t_headers):
       t_row[th] = td.text.replace('\n', '').strip()

    table_data.append(t_row)

print(table_data)
print(table_data[0])
print(table_data[1])
print(table_data[2])
print(table_data[3])
print(table_data[4])
print(table_data[5])
print(table_data[6])
print(table_data[7])
print(table_data[8])
print(table_data[9])

    # Put the data for the table with his heading.


# Step 4: Export the data to csv
    # Create csv file for each table
with open('/Users/anushakonakalla/Desktop/Book4.csv', 'w') as out_file:

        headers = [
            "Method name",
            "Parameters",
            "Scalability",
            "Usecase",
            "Geometry (metric used)"
        ]  # == t_headers
        writer = csv.DictWriter(out_file, headers)
        # write the header
        writer.writeheader()
        for row in table_data:
            if row:
                writer.writerow(row)
