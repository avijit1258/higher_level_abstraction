


class Calculator:
    """ This class calculates sum, sub, div, mul operation on two given numbers. """ 

    def init(self):
        """ This function welcomes users to the calculator """

        print("Welcome to One Two calculator. ")
        a, b = self.two_number_input()
        self.operations_to_do(a, b)

    
    def operations_to_do(self, a, b):

        print('Please enter signs for operations to do on this two number.')
        print('Add = +, Sub = -, Div = /, Mul = *, Modular = %')
        print('Enter . to stop doing operations')
        while 1:
            sign = input()
            if sign == '+':
                result  = self.add_two_numbers(a, b)
            elif sign == '-':
                result = self.subtract_two_numbers(a, b)
            elif sign == '/':
                result = self.divide_two_numbers(a, b)
            elif sign == '*':
                result = self.multiply_two_numbers(a, b)
            elif sign == '%':
                result = self.mod_two_numbers(a, b)
            elif sign == '.':
                break
            else:
                print("Invalid input")
            
            print( a, ' ', sign , ' ', b, ' = ', result )

        

    def add_two_numbers(self, a, b):
        """ This function adds two numbers """

        return a + b

    def subtract_two_numbers(self, a, b):
        """ This function subtract two numbers """

        return a - b

    def divide_two_numbers(self, a, b):
        """ This function divide two numbers """

        return a / b

    def multiply_two_numbers(self, a, b):
        """ This function multiply two numbers """

        return a * b
    
    def mod_two_numbers(self, a, b):
        """ This function mod two numbers """

        return a % b

    

    def valid_number(self, num):
        """ This function verifies a variable of int type """

        try:
            value = int(num)
            return True
        except ValueError:
            return False


        
    def two_number_input(self):
        """ inputs two number """

        loop_condition = True
        while loop_condition:
            a = input("Please enter valid first number")

            print(self.valid_number(a))
            if self.valid_number(a) :
                loop_condition = False
        
        loop_condition = True
        while loop_condition:
            b = input("Please enter valid second number")
            if self.valid_number(b) :
                loop_condition = False


        return int(a) , int(b)


    
    

c = Calculator()

c.init()