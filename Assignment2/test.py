from io import BufferedReader
from turtle import color


class Dog:
    animal = 'dog'

    def __init__(self, breed, color):
        self.breed = breed
        self.color = color
        self.color = None

Rodger = Dog('pug','brown')
buzo = Dog('pom','red' )

print(Rodger.color)
