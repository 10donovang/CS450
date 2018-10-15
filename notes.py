import math
print("Inital")
print(-(3/7)*math.log2(3/7)-(4/7)*math.log2(4/7))
print("Star Actors Yes Set1")
print(-(2/3)*math.log2(2/3)-(1/3)*math.log2(1/3))
print("Star Actors No Set1")
print(-(1/2)*math.log2(1/2)-(1/2)*math.log2(1/2))
print("Type Comedy set1")
print(-(1/4)*math.log2(1/4)-(3/4)*math.log2(3/4))
print("Type Drama Set1")
print(-(1/3)*math.log2(1/3)-(2/3)*math.log2(2/3))
print("Plot Shallow set1")
print(-(2/3)*math.log2(2/3)-(1/3)*math.log2(1/3))
print("Plot Deep Set1")
print(-(1/2)*math.log2(1/2)-(1/2)*math.log2(1/2))

# THis is enthropy formula. If there are more than two variables in the end we just add another log2 to the problem.
#If there is more to the branch then we would add another group of entropy. After we caculate each enthropy from the group, we weight them together.  
