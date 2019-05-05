import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
from skimage.exposure import histogram
from skimage import exposure
import skimage
from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from scipy import optimize
from sklearn.cluster import MeanShift


###bild laden
import os

filename = os.path.join('bild7_20190403.bmp')
from skimage import io

bild = io.imread('bild7_20190403.bmp')
print(bild.shape)
plt.gray()

plt.figure(1)
plt.imshow(bild)
plt.close()

####bild drehen
bildrotate = skimage.transform.rotate(bild, -1.55, resize=False, center=None)

plt.figure(2)
plt.imshow(bildrotate)
plt.close()

###bildschneiden

bildROI = bildrotate[409:1255, 1963:2174]
print(bildROI.shape)

bilduntergrund = bildrotate[409:1255, 1618:1829]
plt.figure(6)
plt.imshow(bilduntergrund)
plt.close()

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

###Linie anzeigen
# line_x1 = 126
# plt.axvline(x = line_x1, lw=1)

#####Lineprofile
##erstes rechteck
lineprofile1 = skimage.measure.profile_line(bildROI, (recty1, rectx11), (recty2, rectx21), linewidth=breite)
plt.figure(4)
plt.plot(lineprofile1, lw=1)
plt.ylim(0, 1)
plt.title('Lineprofile1')
plt.close()

##zweites rechteck
lineprofile2 = skimage.measure.profile_line(bildROI, (recty1, rectx12), (recty2, rectx22), linewidth=breite)
plt.figure(5)
plt.plot(lineprofile2, lw=1)
plt.ylim(0, 1)
plt.title('Lineprofile2')
plt.close()
###zusammen

lineprofile = lineprofile1 + lineprofile2
plt.figure(9)
plt.plot(lineprofile, lw=1)
plt.ylim(0, 1.1)
plt.title('lineprofile')
plt.close()

###untergrund
lineprofileunter = skimage.measure.profile_line(bilduntergrund, (5, 90), (830, 111))
plt.figure(8)
plt.plot(lineprofileunter, lw=1)
plt.close()


# print("Linienprofile:\n")
# print(lineprofileunter)


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

plt.figure(10)
plt.plot(x, y, label='data', lw=1)
plt.plot(x, poly_function(x, *popt), 'k-', label='fit')
plt.legend()


####Gaussfit

def gaus1_function(x, mu, sigma, amp, bkg):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg


y = lineprofile
bkg = poly_function
xi = [x for x in range(len(y))]
x = np.asarray(xi)
dy = np.sqrt(y)

params = ['mu', 'sigma', 'amp', 'bkg']
popt, pcov = curve_fit(gaus1_function, x, y, sigma=dy, p0=None)
perr = np.sqrt(np.diagonal(pcov))
for i in range(len(params)):
    print('%16s = % f +- %f' % (params[i], popt[i], perr[i]))
print('')
chi_sq = np.sum(((y - gaus1_function(x, *popt)) / dy) ** 2)
ndof = len(y) - len(popt)
chi_sq_red = chi_sq / ndof
chi_sq_red_err = np.sqrt(2 / ndof)

res = (y - gaus1_function(x, *popt)) / np.sqrt(y)
print('chi^2 = %f, ndof = %d, chi^2_red = %f +- %f' % (chi_sq, ndof, chi_sq_red, chi_sq_red_err))

plt.figure(11)
plt.plot(x, y, label='data', lw=1)
plt.legend()



####minima suchen
minima = argrelmin(lineprofile, order=5)[0]
minliste = list(minima)
print(minima)


slices=[]
for i in range(len(minliste)-1):
    von=minliste[i]
    bis=minliste[i+1]
    slice=(von, bis)
    slices.append(slice)
print(slices)


for von, bis in slices:
    line_place_sclice=lineprofile[von:bis]


    def gaus1_function(x, mu, sigma, amp, bkg):
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bkg

    y = line_place_sclice
    bkg = poly_function
    xi = [x for x in range(len(y))]
    xi= [x - (len(y)/2) for x in range(len(y))]
    x = np.asarray(xi)
    dy = np.sqrt(y)
    print(x)

    params = ['mu', 'sigma', 'amp', 'bkg']
    popt, pcov = curve_fit(gaus1_function, x, y, sigma=dy, p0=None)
    perr = np.sqrt(np.diagonal(pcov))
    for i in range(len(params)):
        print('%16s = % f +- %f' % (params[i], popt[i], perr[i]))
    print('')
    chi_sq = np.sum(((y - gaus1_function(x, *popt)) / dy) ** 2)
    ndof = len(y) - len(popt)
    chi_sq_red = chi_sq / ndof
    chi_sq_red_err = np.sqrt(2 / ndof)

    res = (y - gaus1_function(x, *popt)) / np.sqrt(y)
    print('chi^2 = %f, ndof = %d, chi^2_red = %f +- %f' % (chi_sq, ndof, chi_sq_red, chi_sq_red_err))


    x_verschoben= [a + von+ int(len(y)/2)  for a in x]
    plt.plot(x_verschoben, y,  lw=1)
    plt.plot(x_verschoben, gaus1_function(x, *popt), 'k-')
    plt.legend()




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






