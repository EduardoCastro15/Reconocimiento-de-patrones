from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import io
import numpy as np
import cv2
from skimage import io

image = cv2.imread('CMA-x1.png')


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = [x, y]
        print(pixel)
        b, g, r = image2[y, x]
        print("{},{},{}".format(b, g, r))
        print(image2[y, x])
        print(list(image2[y, x]))


print(image.shape)
gray = rgb2gray(image)
gray.shape

h, w = gray.shape
print(gray.shape)
print('width:  ', w)
print('height: ', h)
#print('channel:', c)

scale_width = w / gray.shape[1]
scale_height = h / gray.shape[0]
scale = min(scale_width, scale_height)
window_width = int(gray.shape[1] * scale)
window_height = int(gray.shape[0] * scale)

cv2.namedWindow("image")
cv2.resizeWindow('image', window_width, window_height)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0], gray.shape[1])


def maxBoxFilter(n, path_to_image):
    img = cv2.imread(path_to_image)

    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    imgResult = cv2.dilate(img, kernel)
    cv2.imwrite('BN.jpg', imgResult)
    # Shows the result
    # Adjust the window length
    cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL)
    cv2.imshow('Result with n ' + str(n), imgResult)


def minBoxFilter(n, path_to_image):
    img = cv2.imread(path_to_image)

    # Creates the shape of the kernel
    size = (n, n)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    imgResult = cv2.erode(img, kernel)
    cv2.imwrite('BN.jpg', imgResult)
    # Shows the result
    cv2.namedWindow('Result min with n ' + str(n),cv2.WINDOW_NORMAL)  # Adjust the window length
    cv2.imshow('Result min with n ' + str(n), imgResult)


filename = 'BN.jpg'
cv2.imwrite(filename, 255*gray)
maxBoxFilter(3, filename)
minBoxFilter(5, filename)

filename2 = 'boscoso.jpg'
filename3 = 'cieloSuelo.jpg'

image2 = cv2.imread("BN.jpg")

OR = cv2.bitwise_or(image, image2)
cv2.imshow('OR', OR)
cv2.imwrite(filename2, OR)

mask_inv = cv2.bitwise_not(image2)
cv2.imshow('NOT', mask_inv)

OR2 = cv2.bitwise_or(image, mask_inv)
cv2.imshow('OR2', OR2)
cv2.imwrite(filename3, OR2)

# ---- GUARDAR EN ARCHIVOS---------

imageOR = cv2.imread('boscoso.jpg')

lw = [255, 255, 255]

f1 = open("boscosoRGB.txt", "w+")
for i in range(h):
    for j in range(w):
        if list(imageOR[i, j]) != lw:
            #pixel = [i, j]
            # print(pixel)
            #print(imageOR[i, j])
            b = imageOR[i, j][0]
            g = imageOR[i, j][1]
            r = imageOR[i, j][2]
            if r < 240 and g < 240 and b < 240:
                f1.write(str(r)+',')
                f1.write(str(g)+',')
                f1.write(str(b)+'\n')
f1.close()

imageOR2 = cv2.imread('cieloSuelo.jpg')
f2 = open("cieloRGB.txt", "w+")
f3 = open("sueloRGB.txt", "w+")

for i in range(h):
    for j in range(w):
        if list(imageOR2[i, j]) != lw:
            b = imageOR2[i, j][0]
            g = imageOR2[i, j][1]
            r = imageOR2[i, j][2]
            if b > r:
                f2.write(str(r)+',')
                f2.write(str(g)+',')
                f2.write(str(b)+'\n')
            elif r < 240 and g < 240 and b < 240:
                f3.write(str(r)+',')
                f3.write(str(g)+',')
                f3.write(str(b)+'\n')
f2.close()
f3.close()


while(1):
    cv2.imshow("image", image2)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
