d1=((x1-x)**2 + (x2-y)**2)**(1/2)
d2=((x3-x)**2 + (x4-y)**2)**(1/2)
d3=((x5-x)**2 + (x6-y)**2)**(1/2)  

print("Distancia euclidiana con respecto al Z1")
print("d1(x,y)= " + str(d1))
print("Distancia euclidiana con respecto al Z2")
print("d2(x,y)= " + str(d2))
print("Distancia euclidiana con respecto al Z3")
print("d3(x,y)= " + str(d3))

clase = 0
if d1<=d2 and d1<=d3:
  clase = 1
elif d2<=d1 and d2<=d3:
  clase = 2
elif d3<=d1 and d3<=d2:
  clase = 3
print("Clase: " + str(clase))