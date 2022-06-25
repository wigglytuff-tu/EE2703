'''
--------------------------------------------------------------------------------------------------
Assignment 2 - EE2703 (Jan-May 2022)
Purvam Jain (EE20B101)

The following code is able to solve AC(single frequency) and DC circuits with one-port elements.
We use Modified Nodal Analysis to generate coefficient matrix(M1) and Marix(M2) and solve them
for given parameters. We assume first node to be from node and second node as to node respectively.

Acknowledment: https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA6.html --> The following 
resource was referenced to understand MNA algortihm.
--------------------------------------------------------------------------------------------------
'''


# import all necessary libraries
import numpy as np
import sys
import cmath          # For complex numbers
import math

# To improve readability
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"
AC = ".ac"

# from = lower voltage
# to = higher voltage

w = 0       # frequency
is_ac =0    # bool value used later to check if circuit consists of AC component

# Class to extract required details from respective lines in netlist annd fill in MNA Matrix 
class Node:                         
    def __init__(self,name,index):
        self.name = name
        self.index = index
        self.incurrent_passive = []   # Passive refers to r,l,c elements
        self.outcurrent_passive = []
        self.incurrentV = []
        self.outcurrentV = []
        self.incurrent_I = []
        self.outcurrent_I = []
        


# Common Class for Passive Elements R,L,C
class Passive_Elements:
    def __init__(self,name,node1,node2,value,element):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        if(element == 'R'):
            self.value = value
        
        elif(element == 'C'):
            if(is_ac):
                print(w,value)
                self.value = complex(0,-1/(w*value))
                print(self.value)
            else:
                self.value = 1e100   # Active when AC source present
        elif(element == 'L'):
            if(is_ac):
                self.value = complex(0,(w*value))
            else:
                self.value = 1e-100     # Active when AC source present
        self.element = element
    
# Class for Independent Voltage and Current Sources

class IndependentSources():
    def __init__(self,name,node1,node2,value,element):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value
        self.element = element



# Empty Lists created to hold respective values
node =[]        #list of Node Objects
nodes =[]       #list of Node names
resistors = []
capacitors = []
inductors = []
voltage_sources = []
current_sources = []
nodes.append("GND")
dummy = Node("GND",0)
node.append(dummy)
# print("I am printing node\n\n\n\n")
# print(node)

# Function to read file and return the part between .circuit and .end
def file_read():
    global is_ac,w         # To modify gloabal variables
    if(len(sys.argv)!=2):
        print("Invalid number of arguments! Pass the netlist file as the second argument.")
        exit()
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        contains = []
        flag = 0
        for l in lines:
            tokens = l.split()
            # print(tokens)
            if(len(tokens) == 0):
                continue
            if (CIRCUIT_START== tokens[0]):
                flag = 1
                continue
            if flag:
                if (CIRCUIT_END== tokens[0] and (len(tokens)==1 or tokens[1][0] =='#')):
                    flag = 0
                contains.append(l)
            if(AC == tokens[0]  and tokens[1][0] == 'V'):
                is_ac = 1
                w = value_mapper(tokens[2])
                print("Frequency :" , w, is_ac)
                w = w* 2*math.pi
                break
        if(len(contains)==0):
            print("Empty File or missing .circuit flag")
            exit()
    return contains

# Converts string to float
def value_mapper(x):
    y = len(x)
    if(not x[y-1].isalpha()):   # Last index of value
        return float(x)
    if(x[y-1]=='p'):
        return float(x[0:y-1])* 1e-12   
    if(x[y-1]=='n'):
        return float(x[0:y-1])* 1e-9
    if(x[y-1]=='u'):
        return float(x[0:y-1])* 1e-6
    if(x[y-1]=='m'):
        return float(x[0:y-1])* 1e-3
    if(x[y-1]=='k'):
        return float(x[0:y-1])* 1e3
    if(x[y-1]=='M'):
        return float(x[0:y-1])* 1e6
    if(x[y-1]=='G'):
        return float(x[0:y-1])* 1e9  

# Function to append node objects and node names
# We use index of the node as it appears to name it so it doesn't matter if name is alphanumeric 
def append(n1,n2):
    # if(not (n1.isalnum() and n2.isalnum())):
    #     print("Node names are alphanumeric.")
    #     exit()

    if(n1 not in nodes):
        nodes.append(n1)
        dummy = Node(n1,nodes.index(n1))
        node.append(dummy)
        
    if(n2 not in nodes):
        nodes.append(n2)
        dummy = Node(n2,nodes.index(n2))
        node.append(dummy)
    return nodes.index(n1),nodes.index(n2)


# Function that breaks the line down into components, recognises the component and creates an object for the component
def parse_line(line):
    tokens = line.split()
    l = len(tokens)
    if(l==4 or (l>4 and tokens[4][0] =='#') and (tokens[0][0] == 'R' or tokens[0][0] == 'L' or tokens[0][0] == 'C' or tokens[0][0] == 'V' or tokens[0][0] == 'I')):
        element = tokens[0]
        n1 = tokens[1]
        n2 = tokens[2]
        value = tokens[3]
        val = value_mapper(value)
        
        from_node_index,to_node_index = append(n1,n2)

        if(tokens[0][0] == 'R' or tokens[0][0] == 'C' or tokens[0][0] == 'L'):
            x = Passive_Elements(element,from_node_index,to_node_index,val,tokens[0][0])
            node[from_node_index].outcurrent_passive.append(x)
            
            node[to_node_index].incurrent_passive.append(x)
          
            if(tokens[0][0] == 'R'):
                resistors.append(x)            
            if(tokens[0][0] == 'L'):
                inductors.append(x)
            if(tokens[0][0] == 'C'):
                capacitors.append(x) 
        else:
            print("Syntax Error in netlist File")       
        
    elif(l == 6 or (l>6 and tokens[6][0] =='#')):
        if((tokens[0][0] == 'V' or tokens[0][0] == 'I') and tokens[3] == 'ac'):
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            value = value_mapper(tokens[4])
            value/=2
            phase = value_mapper(tokens[5])
            from_node_index,to_node_index = append(n1,n2)

            x = IndependentSources(element,from_node_index,to_node_index,complex(value*math.cos(phase),math.sin(phase)),tokens[0][0])
            if(tokens[0][0] == 'V'):
                voltage_sources.append(x)  
                node[from_node_index].outcurrentV.append(x)  
                node[to_node_index].incurrentV.append(x)        
            if(tokens[0][0] == 'I'):
                current_sources.append(x)
                node[from_node_index].outcurrent_I.append(x)  
                node[to_node_index].incurrent_I.append(x)  
        
        else:
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            n3 = tokens[3]
            n4 = tokens[4]
            value = tokens[5]   

#             if(not (n1.isalnum() and n2.isalnum() and n3.isalnum() and n4.isalnum())):
#                 print("Node names are alphanumeric.")
#                 exit()

    elif(l==5 or (l>5 and tokens[5][0] =='#')):
        if((tokens[0][0] == 'V' or tokens[0][0] == 'I') and tokens[3] == 'dc'):
            if(is_ac):
                print("Error:Multiple frequencies in same circuit")
                exit()
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            value = value_mapper(tokens[4])
            from_node_index,to_node_index = append(n1,n2)

            if(tokens[0][0] == 'V'):
                x = IndependentSources(element,from_node_index,to_node_index,value,tokens[0][0])
                voltage_sources.append(x)  
                node[from_node_index].outcurrentV.append(x)  
                node[to_node_index].incurrentV.append(x)        
            if(tokens[0][0] == 'I'):
                x = IndependentSources(element,from_node_index,to_node_index,value,tokens[0][0])
                current_sources.append(x)
                node[from_node_index].outcurrent_I.append(x)  
                node[to_node_index].incurrent_I.append(x)  
            
        
        else:   
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            V = tokens[3]
            value = tokens[4]
            # if(not (n1.isalnum() and n2.isalnum())):
            #     print("Node names are alphanumeric")

    return

def populate_Matrices():
    if(is_ac==1):
        M1 = np.zeros((len(node)+len(voltage_sources),len(node)+len(voltage_sources)),dtype=np.complex)
        M2 = np.zeros(len(node)+ len(voltage_sources),dtype=np.complex)
    else:
        M1 = np.zeros((len(node)+len(voltage_sources),len(node)+len(voltage_sources)))
        M2 = np.zeros(len(node)+ len(voltage_sources))
    for n in node:
        #dealing with all resistors

        if(n.name == "GND"):
            M1[0][0] = 1
            M2[0] = 0
            continue
        for x in n.incurrent_passive:
            M1[node.index(n)][x.node1] += (1/x.value)
            M1[node.index(n)][node.index(n)] -= (1/x.value)
        for x in n.outcurrent_passive:
            M1[node.index(n)][x.node2] += (1/x.value)
            M1[node.index(n)][node.index(n)] -= (1/x.value)
    
    for x in voltage_sources:

        M1[x.node1][voltage_sources.index(x)+len(node)] -=1
        M1[x.node2][voltage_sources.index(x)+len(node)] +=1
        M1[voltage_sources.index(x)+len(node)][x.node1] -=1    #from =  -ve
        M1[voltage_sources.index(x)+len(node)][x.node2] +=1    #to = +ve
        M2[voltage_sources.index(x)+len(node)] = x.value
        

    for x in current_sources:
        if(is_ac==1):
            M2[x.node1] += x.value/2                  #from = leaving
            M2[x.node2] -= x.value/2                   #to = entering
        else:
            M2[x.node1] += x.value                   #from = leaving
            M2[x.node2] -= x.value
    M1[0][len(node):] = np.zeros(len(voltage_sources))
    M2[0] = 0
    return M1,M2


for l in file_read():
    print(l)
    parse_line(l)

M1,M2 = populate_Matrices()
print(M1)
print(M2)
try:
    X = np.linalg.solve(M1,M2)
except:
    print("Unsolvable Matrix")
    exit()
i=0
for n in nodes:
    print("Voltage at Node " + n+" =  " , X[i])
    i = i+1
for V in voltage_sources:
    print("Current Through Voltage Source " + V.name + " = ", X[i])
    i= i+1

