# Employee class
class Employee:
    countEmployees = 0
    salaries = []

    #Default constructor function
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department

        #appending salaries to the list
        Employee.salaries.append(self.salary)
        Employee.countEmployees = Employee.countEmployees + 1


    #Average salary of all employess
    def avgsalary(self, salaries):
        length = len(salaries)
        print(length)
        totalsalary = 0
        for salary in salaries:
            totalsalary = totalsalary + salary
        print(totalsalary)
        print("Average Salary = ", totalsalary/length)


#Full time Employee class
class FulltimeEmployee(Employee):
    empId = 0
    def __init__(self, empId, name, family, salary, department):
        self.empId = empId
        print(self.empId)
        Employee.__init__(self, name, family, salary, department)
        FulltimeEmployee.empId = FulltimeEmployee.empId + 1


Employee1 = Employee( "abc", "def", 5000, "CSE")
Employee2 = Employee( "ghi", "def", 8000, "ECE")
Employee3 = Employee( "jkl", "def", 8000, "MPharmcy")
FulltimeEmployee1 = FulltimeEmployee(157, "mno", "def", 10000, "ECE",)
FulltimeEmployee2 = FulltimeEmployee(289, "pqr", "def", 15000, "civil")

print("Employee1 Name is: ", Employee1.name)
print("Employee2 Name is: ", Employee2.name)
print("Employee3 Name is: ", Employee3.name)
print("Fulltime Employee1 Name is: ", FulltimeEmployee1.name)
print("Fulltime Employee2 Name is: ", FulltimeEmployee2.name)

# Access data member using FulltimeEmployee class
print("Number of Employees: ", Employee.countEmployees)
print("Number of Fulltimeemployees:", FulltimeEmployee.empId)

FulltimeEmployee1.avgsalary(FulltimeEmployee.salaries)
