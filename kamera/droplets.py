import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import exposure
import skimage
import os
from skimage import io
from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from scipy import special


#####comment

###bild laden
filename = os.path.join("C:\\Users\\Lana\\Desktop\\ba", 'Bild43_20190403.bmp')
bild = io.imread("C:\\Users\\Lana\\Desktop\\ba", 'Bild43_20190403.bmp')
print(bild.shape)
plt.gray()

plt.figure(1)
plt.imshow(bild)
plt.close()

####bild drehen
bildrotate = skimage.transform.rotate(bild, -1.2, resize=False, center=None)

plt.figure(2)
plt.imshow(bildrotate)
plt.close()

###bildschneiden
bildROI = bildrotate[50:588, 688:777]  # [y_start:y_stopp, x_start:x_stopp]
# für bild 43, 44 [50:588, 688:777]
print(bildROI.shape)
plt.figure(3)
plt.imshow(bildROI)
plt.close()

##untergrund
untergrund = bildrotate[50:588, 777:966]
plt.figure(4)
plt.imshow(untergrund)
plt.close()

###rechteck anzeigen

fig, ax = plt.subplots(1)
ax.imshow(bildROI)
rectx1 = 29
recty1 = 10
breite = 26
länge = 510
rectx2 = rectx1 + breite
recty2 = recty1 + länge

rect1 = patches.Rectangle((rectx1, recty1), breite, länge, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect1)

plt.figure(5)
plt.imshow(bildROI)
plt.close()

#####Lineprofile

lineprofile = skimage.measure.profile_line(bildROI, (recty1, rectx1), (recty2, rectx2), linewidth=breite)
plt.figure(6)
plt.plot(lineprofile, lw=1)
plt.title('Lineprofile')
# plt.close()

##linienprofil untergrund
lineprofileunter = skimage.measure.profile_line(untergrund, (10, 29), (520, 55), linewidth=26)
plt.figure(7)
plt.plot(lineprofileunter, lw=1)
plt.close()


###polynom fit untergrund

def poly_function(x, a, b, c, d):
    return a * x + b * x ** 2 + c * x ** 3 + d


y = lineprofileunter
xi = [x for x in range(len(y))]
x = np.asarray(xi)
dy = np.sqrt(y)

params = ['a', 'b', 'c', 'd']
popt, pcov = curve_fit(poly_function, x, y, sigma=dy, p0=None)
perr = np.sqrt(np.diagonal(pcov))
for i in range(len(params)):
    print('%16s = % f +- %f' % (params[i], popt[i], perr[i]))
print('')

chi_sq = np.sum(((y - poly_function(x, *popt)) / dy) ** 2)
ndof = len(y) - len(popt)
chi_sq_red = chi_sq / ndof
chi_sq_red_err = np.sqrt(2 / ndof)

res = (y - poly_function(x, *popt)) / np.sqrt(y)
print('chi^2 = %f, ndof = %d, chi^2_red = %f +- %f' % (chi_sq, ndof, chi_sq_red, chi_sq_red_err))

plt.figure(8)
plt.errorbar(x, y, label='data', lw=1)
plt.plot(x, poly_function(x, *popt), 'k-', label='fit')
plt.legend()

# Faltung rechteck und gauss fit

# irgendwie noch ander fehlerfunktion
x = lineprofile


def faltung(x, R, s):
    return 1 / 2 * (1 - special.erfc((x - R) / s))


params = ['s', 'R']
popt, pcov = curve_fit(faltung, x, y, sigma=dy, p0=None)
perr = np.sqrt(np.diagonal(pcov))
for i in range(len(params)):
    print('%16s = % f +- %f' % (params[i], popt[i], perr[i]))
print('')

chi_sq = np.sum(((y - faltung(x, *popt)) / dy) ** 2)
ndof = len(y) - len(popt)
chi_sq_red = chi_sq / ndof
chi_sq_red_err = np.sqrt(2 / ndof)

res = (y - faltung(x, *popt)) / np.sqrt(y)
print('chi^2 = %f, ndof = %d, chi^2_red = %f +- %f' % (chi_sq, ndof, chi_sq_red, chi_sq_red_err))

plt.figure(9)
plt.errorbar(x, y, label='data', lw=1)
plt.plot(x, faltung(x, *popt), 'k-', label='fit')
plt.legend()



####minima suchen
minima = argrelmin(lineprofile, order=25)[0]
minliste = list(minima)
print(minima)

####abstand messen
min = minliste[1:]
abstand = []
for x in range(len(minliste) - 1):
    abstand1 = min[x] - minliste[x]
    abstand.append(abstand1)
print("abstand :", abstand)

# abstand_2 = [ a -b for a, b in zip(minliste[1:],minliste[:-1])]
# print(abstand_2)

###mittelwert

mittelwert = np.mean(abstand)
print("mittelwert : ", mittelwert)

plt.show()
