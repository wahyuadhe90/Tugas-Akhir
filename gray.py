import cv2
from matplotlib import pyplot as plt

## Read as BGR
img = cv2.imread("1.jpg")
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

imgplot = plt.imshow(gray)
plt.title("Hasil Segmentasi")
plt.show()
cv2.imwrite("12.jpg",gray)
