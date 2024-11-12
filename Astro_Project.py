# importing the image
import numpy as np
from astropy.io import fits
from photutils.detection import IRAFStarFinder as iraf
import matplotlib.pyplot as plt

fits_data = fits.getdata('M45_Final.FTS')
sectionR = fits_data[0,:,:]
sectionG = fits_data[1,:,:]
sectionB = fits_data[2,:,:]

iraffind = iraf(fwhm = 4.0, threshold = 1e+4, roundlo=-1.0, roundhi=1.0)
sources = iraffind(sectionR-np.median(sectionR))

def photometry(section,sources,r_ap,r_in,r_out,exptime,zeropoint,color):

    # Importing needed libraries
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from photutils.aperture import CircularAnnulus as CircAn, CircularAperture as CircAp, ApertureStats as ApStats, aperture_photometry as ApPho

    # Finding the stars
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
    aperstats = ApStats(section, annulus_aperture)
    bkg_mean = aperstats.mean
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
    error = aperstats.std/(np.sqrt(len(aperstats)))
    star_data['magnitudes'] = mags
    star_data.pprint()
    print('')
    return mags, fluxes, error

# Establishing the magnitudes, calculating error and finding 3 sigma upper limit 
I_mag, I_fluxes, uncI = photometry(section=sectionR, sources=sources, r_ap=20.0,r_in=22,r_out=30,exptime=30,zeropoint=26.14,color='red')
V_mag, V_fluxes, uncV = photometry(section=sectionG, sources=sources, r_ap=20.0,r_in=22,r_out=30,exptime=30,zeropoint=26.14,color='green')
B_mag, B_fluxes, uncB = photometry(section=sectionB, sources=sources, r_ap=20.0,r_in=22,r_out=30,exptime=30,zeropoint=26.14,color='blue')

uncI_mag = ((2.5/np.array(I_fluxes))*uncI)
uncV_mag = ((2.5/np.array(V_fluxes))*uncV)
uncB_mag = ((2.5/np.array(B_fluxes))*uncB)

I_sul = np.mean(I_mag) + 3*np.std(I_mag)
V_sul = np.mean(V_mag) + 3*np.std(V_mag)
B_sul = np.mean(B_mag) + 3*np.std(B_mag)
print('Three sigma upper limit I = ', I_sul, 'Three sigma upper limit V = ', V_sul, 'Three sigma upper limit B = ', B_sul)

BV = np.subtract(B_mag,V_mag)
diagram, (I_mags,V_mags,B_mags) = plt.subplots(1,3,sharey=True)

I_mags.set_xlabel('B-V')
I_mags.set_ylabel('I magnitude')
I_mags.invert_yaxis()
I_mags.errorbar(BV,I_mag,xerr=(np.array(uncB_mag)+np.array(uncV_mag)),yerr=uncI_mag,ls='none', color='purple')
I_mags.plot(BV,I_mag,'o',color='Red')

V_mags.set_xlabel('B-V')
V_mags.set_ylabel('V magnitude')
V_mags.invert_yaxis()
V_mags.errorbar(BV,V_mag,xerr=(np.array(uncB_mag)+np.array(uncV_mag)),yerr=uncV_mag,ls='none', color='purple')
V_mags.plot(BV,V_mag,'o',color='Green')

B_mags.set_xlabel('B-V')
B_mags.set_ylabel('B magnitude')
B_mags.invert_yaxis()
B_mags.errorbar(BV,B_mag,xerr=(np.array(uncB_mag)+np.array(uncV_mag)),yerr=uncB_mag,ls='none', color='purple')
B_mags.plot(BV,B_mag,'o',color='Blue')

plt.show()