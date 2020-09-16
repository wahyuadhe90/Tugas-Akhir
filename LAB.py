import cv2 as cv2
from matplotlib import pyplot as plt

img = cv2.imread("1.jpg")

r = img[:,:,0]/255
g = img[:,:,1]/255
b = img[:,:,2]/255

def rgbtoxyz(rgb):

    for value in [r, g, b]:
        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value /= 12.92

    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505

    X = float (X) / 95.047  
    Y = float (Y) / 100.0  
    Z = float (Z) / 108.883

    return XYZ

for value in XYZ:
        if value > 0.008856:
            value = value ** (float(1 / 3))
        else:
            value = (7.787 * value) + (16 / 116)

L = (116 * X) - 16
a = 500 * (X - Y)
b = 200 * (Y - Z)

    return Lab

