from scipy.optimize import leastsq
from skimage import io
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com
import numpy as np
import matplotlib.pyplot as plt

# calculate radial distances from a given point (x0,y0)
def radial_dist ( x_size, y_size, x_center, y_center):
    x = np.arange(x_size) - x_center
    y1 = np.ones(y_size)
    dx = np.outer(y1,x)**2

    y = np.arange(y_size) - y_center
    x1 = np.ones(x_size)
    dy = np.outer(y,x1)**2
    d = np.sqrt(dx + dy)

    return d

# For a given image data and source position (xc,yc),
# create a radial profile plot with an inset of the source
def radplot(im,xc0,yc0,boxsize=30,fwhm=4.0):
    # make a cutout of boxsize x boxsize
    imcut = Cutout2D(im,[xc0,yc0],boxsize)
    # let&#39;s do a more precise source position estimation
    xc,yc = centroid_com(imcut.data)

    y_size,x_size = imcut.data.shape
    d2 = radial_dist(x_size, y_size, xc, yc)

    iarr = imcut.data.reshape(-1)
    darr = d2.reshape(-1)

    dfull = np.concatenate( (-darr[::-1],darr) )
    ifull = np.concatenate( ( iarr[::-1],iarr) )

    # Fit with a gaussian
    fitfunc = lambda p, x: p[0]*np.exp( -(x-p[1])**2/(2.0*p[2]**2)) + p[3]
    errfunc = lambda p, x, y: fitfunc(p,x) - y

    pinit = [ifull.max(),0.0,fwhm/2.3548,np.median(iarr)]
    pfinal, success = leastsq(errfunc,pinit,args=(dfull,ifull))
    fit = fitfunc(pfinal,darr)

    return darr,iarr,fit,pfinal,imcut

if __name__ == '__main__':
    filename = 'M45_4.DNG'
    img = io.imread(filename)
    im0 = img[:,:,0] # take only the first color channel as a test
    yc, xc = 1863, 1838 # approx position of the source in the main image
    darr,iarr,fit,pfinal,imcut = radplot(im0, xc, yc)
    from matplotlib.colors import LogNorm
    plt.figure()
    plt.imshow(im0,cmap='grey',norm=LogNorm())

    # Make a plot
    fig, ax = plt.subplots()

    ax.plot(darr,iarr,'r.',ms=1.0)
    ax.plot(darr,fit,'k:')
    ax.set_title(f'Source at {xc,yc} in {filename}')
    ax.set_xlabel('Distance from (%.1f,%.1f) in pixels' % (xc,yc))
    ax.set_ylabel('Data Number')
    ax.axvline(pfinal[2]*2.3548*0.5)
    ax.text(0.25,0.5,'HWHM = {pfinal[2]*2.3548*0.5:.2f} pixels',transform=ax.transAxes)
    ax_inset = ax.inset_axes([0.6,0.6,0.35,0.35])
    ax_inset.plot(xc,yc,'r+',ms=15.0)
    ax_inset.imshow(imcut.data,aspect='equal', origin='lower')
    ax_inset.get_xaxis().set_visible(False)
    ax_inset.get_yaxis().set_visible(False)
    plt.show()