import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
from skimage import exposure
import skimage
from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from scipy import optimize



###bild laden
import os

filename = os.path.join('bild7_20190403.bmp')
from skimage import io

bild = io.imread('bild7_20190403.bmp')
print(bild.shape)
plt.gray()

####bild drehen
bildrotate = skimage.transform.rotate(bild, -1.55, resize=False, center=None)

###bildschneiden

bildROI = bildrotate[409:1255, 1963:2174]
print(bildROI.shape)

bilduntergrund = bildrotate[409:1255, 1618:1829]


###rechteck anzeigen

fig, ax = plt.subplots(1)
ax.imshow(bildROI)
rectx11 = 90
rectx12 = 117
recty1 = 5
breite = 21
rectx21 = rectx11 + breite
rectx22 = rectx21 + breite
länge = 830
recty2 = recty1 + länge

rect1 = patches.Rectangle((rectx11, recty1), breite, länge, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect1)
rect2 = patches.Rectangle((rectx12, recty1), breite, länge, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect2)

plt.figure(3)
plt.imshow(bildROI)
plt.close()
plt.show()