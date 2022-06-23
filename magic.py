### Understanding Magic Methods

#Packages
import os
import sys
import pandas as pd
import numpy as np

# ex1
class Car:
    def __init__(self):
        print("자동차 객체 생성됨")

car = Car()

# ex2
class Stock:
    pass

a = Stock()
b = Stock()
print(a+b)

# ex3
class MyFunc:
    def __init__(self):
        print("__init__이 호출됬어요")
    def __call__(self, *args, **kwargs):
        print("__call__이 호출됬어요")
f = MyFunc()
f()
