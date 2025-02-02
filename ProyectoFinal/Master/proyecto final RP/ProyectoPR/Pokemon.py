from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from functools import partial

global w1, r1, r2, e1, e2, e3, e4, e5, e6

def im_show(pic,titulo = "",ip = "nearest"):
    fig, axs = plt.subplots(1, figsize=(30,5))
    axs.imshow(pic,cmap='gray')
    axs.set_xlabel(titulo)
    plt.show()
    return

def grises(pic):
  height=pic.shape[0]
  width=pic.shape[1]
  channels=1
  if pic.ndim==3:
    channels=pic.shape[2]
  if channels==1:
    return pic
  factor=np.array([0.24,0.65,0.11])
  if channels==4:
    factor=np.array([0.24,0.65,0.11,1])
  pic2=np.dot(pic.astype(dtype=np.float),factor)
  return pic2.astype(dtype=np.uint8)

def grises2(img,color = 0):
  alto = img.shape[0]
  ancho = img.shape[1]
  canal = np.zeros((alto,ancho))
  for i in range(alto):
    for j in range(ancho):
      canal[i][j] = img[i][j][color]
  return canal.astype(dtype=np.uint8)

def noise(img, salt, pepper):
    height=img.shape[0]
    width=img.shape[1]  
    img_r=np.asarray(img.copy(),order="C")
    hw=height*width
    if salt>0 and salt<=1:
        npixels=int(float(hw)*salt)
        for i in range(npixels):
            x = np.random.randint(0,width,1)
            y = np.random.randint(0,height,1)
            img_r[y[0],x[0]]=255
    if pepper>0 and pepper<=1:
        npixels=int(float(hw)*pepper)
        for i in range(npixels):
            x = np.random.randint(0,width,1)
            y = np.random.randint(0,height,1)
            img_r[y[0],x[0]]=0
    return img_r

def unificar(lista):
    return [[item] for sublista in lista for item in sublista]


def morfologica_max_aprendizaje(x,y):
  M = np.zeros((y[0].shape[0],x[0].shape[0]))
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      maximo = y[0][i][0] - x[0][j][0]
      for p in range(len(x)):
        if (y[p][i][0] - x[p][j][0]) > maximo :
          maximo = y[p][i][0] - x[p][j][0]
      M[i][j] = maximo
  return M

def morfologica_max_recuperacion(x,M,yE,etiqueta):
  y = np.zeros((M.shape[0],1))
  for i in range(M.shape[0]):
    minimo = M[i][0] + x[0][0]
    for j in range(x.shape[0]):
      if (M[i][j] + x[j][0]) < minimo :
        minimo = M[i][j] + x[j][0]
    y[i][0] = minimo
  
  for indice in range(len(etiqueta)):
    if (y == yE[indice]).all():
      return("y:" + str(y) + "\n\nEs un ",etiqueta[indice].split(".")[0])
      
  print("No encontrado")

def morfologica_min_aprendizaje(x,y):
  W = np.zeros((y[0].shape[0],x[0].shape[0]))
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      minimo = y[0][i][0] - x[0][j][0]
      for p in range(len(x)):
        if (y[p][i][0] - x[p][j][0]) < minimo :
          minimo = y[p][i][0] - x[p][j][0]
      W[i][j] = minimo
  return W

def morfologica_min_recuperacion(x,W,yE,etiqueta):
   
  y = np.zeros((W.shape[0],1))
  for i in range(W.shape[0]):
    maximo = W[i][0] + x[0][0]
    for j in range(x.shape[0]):
      if (W[i][j] + x[j][0]) > maximo :
        maximo = W[i][j] + x[j][0]
    y[i][0] = maximo
  
  for indice in range(len(etiqueta)):
    if (y == yE[indice]).all():
      return("y:" + str(y) + "\n\nEs un " + etiqueta[indice].split(".")[0])
      
  return("No encontrado")

pokemones = ["Charmander.bmp","Gengar.bmp","Mewtwo.bmp","Pikachu.bmp","Squirtle.bmp"]
y_array = [np.asarray([[1],[0],[0],[0],[0]]),np.asarray([[0],[1],[0],[0],[0]]),np.asarray([[0],[0],[1],[0],[0]]),np.asarray([[0],[0],[0],[1],[0]]),np.asarray([[0],[0],[0],[0],[1]])]
x_array = [np.asarray(unificar(grises2(np.asarray(Image.open(pokemon))).tolist())) for pokemon in pokemones]
#x = [np.asarray([[1],[0],[1],[0],[1]]),np.asarray([[1],[1],[0],[0],[1]]),np.asarray([[1],[0],[1],[1],[0]])]
#y = [np.asarray([[1],[0],[0]]),np.asarray([[0],[1],[0]]),np.asarray([[0],[0],[1]])]
M = morfologica_max_aprendizaje(x_array,y_array)

W = morfologica_min_aprendizaje(x_array,y_array)



def showInvMult(varInv):
    tipo = e1.get()
    pokemon = e2.get()
    salt = float(e3.get())
    pepper = float(e4.get())
    imagen = noise(grises2(np.asarray(Image.open(pokemon))),salt,pepper)    
    
    patron = np.asarray(unificar(imagen.tolist()))
    if tipo in "min":
        r1 = morfologica_min_recuperacion(patron,W,y_array,pokemones)
        pass
    else:
        r1 = morfologica_max_recuperacion(patron,M,y_array,pokemones)
        pass
    varInv.set(r1)
    img = Image.fromarray(imagen)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(w1, image = img)
    panel.image = img 
    panel.place(x=140,y=450)
    
    


x=400
y=630
title = "Memoria heteroasociativa V1.0"
w1 = Tk()

w1.title(title)
w1.maxsize(x,y)

l0 = Label(w1, text="Memoria heteroasociativa V1.0 by Isaac Ortiz y Ramses Nuñez", width=50, bg='BLACK', fg='#01DF3A')
l0.place(x=15,y=2)

l1 = Label(w1, text="Ingrese 'min' o 'max':", width=20, bg='BLACK', fg='#01DF3A')
l1.place(x=50,y=40)
e1 = Entry(w1, bd=1, width=5, bg='BLACK', fg='#01DF3A', justify=LEFT)
e1.place(x=260, y=40)


l2 = Label(w1, text="Ingrese el nombre de la imagen:", width=28, bg='BLACK', fg='#01DF3A')
l2.place(x=50,y=70)
e2 = Entry(w1, bd=1, width=20, bg='BLACK', fg='#01DF3A', justify=LEFT)
e2.place(x=260, y=70)


l3 = Label(w1, text="Ingrese la cantidad de ruido sal: ", width=28, bg='BLACK', fg='#01DF3A')
l3.place(x=50,y=100)
e3 = Entry(w1, bd=1, width=10, bg='BLACK', fg='#01DF3A', justify=LEFT)
e3.place(x=260, y=100)

l4 = Label(w1, text="Ingrese la cantidad de ruido pimienta: ", width=33, bg='BLACK', fg='#01DF3A')
l4.place(x=49,y=130)
e4 = Entry(w1, bd=1, width=10, bg='BLACK', fg='#01DF3A', justify=LEFT)
e4.place(x=280, y=130)

varInv = StringVar()
l5 = Label(w1, textvariable=varInv, bg='BLACK', fg='#01DF3A')
l5.place(x=150,y=300)
    
action_InvMult = partial(showInvMult, varInv)
b1 = Button(w1, text="ACEPTAR", bg='BLACK', fg='#01DF3A', command = action_InvMult)
b1.place(x=170, y=160)

l6 = Label(w1, text="M:\n" + str(M), width=30, bg='BLACK', fg='#01DF3A')
l6.place(x=10,y=190)

l7 = Label(w1, text="W:\n" + str(W), width=30, bg='BLACK', fg='#01DF3A')
l7.place(x=180,y=190)


w1.configure(bg='Black')
w1.geometry(str(x) + "x" + str(y))
w1.mainloop()