#coidng:utf8
from numpy import *
from aprior import *

mushData = [line.split() for line in open('mushroom.dat').readlines()]

L,suppData = apriori(mushData,minSupport=0.3)
print L
print suppData