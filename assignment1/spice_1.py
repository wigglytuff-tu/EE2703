"""
    EE2703: Applied Programming Lab - 2021
        Assignment 1: Spice - Part 1
            Soham Roy - EE20B130
"""

# for command line arguments and exiting
from sys import argv, exit

# constants that enclose the circuit block
CIRCUIT = ".circuit"
END = ".end"

# to represent each element of the circuit block
class Element:
    def __init__(self, words):
        self.rwords = reversed(words)
        self.name = words[0]
        self.type = self.name[0]

        self.n1 = words[1]
        self.n2 = words[2]
        self.value = words[-1]

        if self.type == "E" or self.type == "F":
            self.n3 = words[3]
            self.n4 = words[4]
        elif self.type == "G" or self.type == "H":
            self.name2 = words[3]

    def __str__(self):
        return " ".join(self.rwords)


# check if the user has given the correct number of arguments
if len(argv) != 2:
    print(f"\nUsage: {argv[0]} <inputfile>")
    exit()

elements = []  # to store all the elements of the circuit

# try to read the file
try:
    with open(argv[1]) as file:
        started = finished = False

        for line in file:
            tokens = line.partition("#")[0].split()  # ignore comments & split the line
            if len(tokens) > 0:
                if started:
                    if tokens[0] == END:
                        finished = True
                        break
                    else:
                        elements.append(Element(tokens))
                elif tokens[0] == CIRCUIT:
                    started = True

    if not finished:
        print("Invalid circuit definition")
        exit(0)

except IOError:
    print("Invalid file")
    exit()

print(*reversed(elements), sep="\n")
