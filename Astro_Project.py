# importing the image
import numpy as np
from astropy.io import fits
from photutils.detection import IRAFStarFinder as iraf
import matplotlib.pyplot as plt

fits_data, header = fits.getdata('M45_final.FTS', header=True)
sectionR = fits_data[0,:,:]     # can adjust these to get certain regions of the image
sectionG = fits_data[1,:,:]
sectionB = fits_data[2,:,:]

iraffind = iraf(fwhm = 3.0, threshold = 1e-5, roundlo=-1.0, roundhi=1.0)
sources = iraffind(sectionR-np.median(sectionR))

def photometry(section,sources,r_ap,r_in,r_out,exptime,zeropoint,color):

    from matplotlib.colors import LogNorm
    from photutils.aperture import CircularAnnulus as CircAn, CircularAperture as CircAp, ApertureStats as ApStats, aperture_photometry as ApPho

    # Creating the apertures and establishing the star positions
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircAp(positions, r=r_ap)
    annulus_aperture = CircAn(positions, r_in = r_in, r_out = r_out)

    plt.figure()
    plt.imshow(section, origin = 'lower', norm = LogNorm(), cmap = 'Greys' )
    apertures.plot(color = 'green', lw = 1.5, alpha = 0.5)
    annulus_aperture.plot(color = color, lw = 1.5, alpha = 0.5)
    
    # Background
    bkgstats = ApStats(section, annulus_aperture)
    apstats = ApStats(section, apertures)
    bkg_mean = bkgstats.mean
    aperture_area = apertures.area_overlap(section)
    total_bkg = bkg_mean * aperture_area
    star_data = ApPho(section, apertures)
    star_data['total_bkg'] = total_bkg
    for col in star_data.colnames:
        star_data[col].info.format = '%.8g'

    # Converting to magnitudes & calculating uncertainty
    mags = []
    fluxes = []
    for line in star_data:
        flux = (abs(line[3]-line[4])/exptime)
        fluxes.append(flux)
        mags.append((zeropoint-2.5*np.log10(flux)))
    error = apstats.std/(np.sqrt(len(apstats)))
    unc_mag = (((2.5/np.array(fluxes)))*error)
    sul = np.mean(mags) + 3*np.std(mags)
    star_data['magnitudes'] = mags
    star_data.pprint()
    print('')
    return mags, unc_mag, sul

# Establishing the magnitudes, calculating magnitude error and finding 3 sigma upper limit 
I_mag, uncI, I_sul = photometry(section=sectionR, sources=sources, r_ap=20.0,r_in=28,r_out=38,exptime=header['exptime'],zeropoint=0,color='red')
V_mag, uncV, V_sul = photometry(section=sectionG, sources=sources, r_ap=20.0,r_in=28,r_out=38,exptime=header['exptime'],zeropoint=0,color='green')
B_mag, uncB, B_sul = photometry(section=sectionB, sources=sources, r_ap=20.0,r_in=28,r_out=38,exptime=header['exptime'],zeropoint=0,color='blue')

print('Three sigma upper limit I = ', I_sul, 'Three sigma upper limit V = ', V_sul, 'Three sigma upper limit B = ', B_sul)

BV = np.subtract(B_mag,V_mag)
diagram, (I_mags,V_mags,B_mags) = plt.subplots(1,3,sharey=True)

def subplot(name,xlabel,ylabel,x,y,xerr,yerr,errcolor,plotcolor):   # defines a function to create the subplots
    name.set_xlabel(xlabel)
    name.set_ylabel(ylabel)
    name.invert_yaxis()
    name.errorbar(x,y,xerr=xerr,yerr=yerr,ls='none', color=errcolor)
    name.plot(x,y,'o',color=plotcolor)

subplot(name=I_mags,xlabel='B-V',ylabel='I magnitude',x=BV,y=I_mag,xerr=(np.array(uncB)+np.array(uncV)),yerr=uncI,errcolor='purple',plotcolor='Red')
subplot(name=V_mags,xlabel='B-V',ylabel='V magnitude',x=BV,y=V_mag,xerr=(np.array(uncB)+np.array(uncV)),yerr=uncV,errcolor='purple',plotcolor='Green')
subplot(name=B_mags,xlabel='B-V',ylabel='B magnitude',x=BV,y=B_mag,xerr=(np.array(uncB)+np.array(uncV)),yerr=uncB,errcolor='purple',plotcolor='Blue')
plt.show()