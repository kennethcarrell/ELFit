import warnings
from astropy.utils.exceptions import AstropyWarning
import os
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.nddata import VarianceUncertainty
from specutils.analysis import centroid, equivalent_width
from specutils.spectra import Spectrum1D, SpectralRegion
#from specutils.fitting import fit_generic_continuum
from astropy import units as u
import matplotlib.pyplot as plt
#warnings.simplefilter('ignore', UserWarning)
#warnings.simplefilter('ignore', category=AstropyWarning)
from ELFit.line_analysis import *

#_MAXAPS_ = 54
#_MAXPIX_ = 2048
## Values for the 82" SES
_NAPS82_ = 17
_NPIX82_ = 1198
## Values for the 107" Coude
_NAPS107_ = 54
_NPIX107_ = 2048


## Open an input file
def retrieveInput(fname):
    specin = fits.open(fname)
    # get telescope info from header
    if(specin[0].header['TELESCOP'] == "mcd107x"):
        NAPS = _NAPS107_
        NPIX = _NPIX107_
        NWAT = 61
    elif(specin[0].header['TELESCOP'] == "mcd82x"):
        NAPS = _NAPS82_
        NPIX = _NPIX82_
        NWAT = 20
    else:
        print("UNKNOWN TELESCOPE")
        return 0
    # get other info from header
    NAME = specin[0].header['OBJECT']
    try:
        HJD = specin[0].header['HJD']
        VHELIO = specin[0].header['VHELIO']
    except:
        HJD = -9999.
        VHELIO = -9999.
    # get the wavelength solution
    soln = ''
    for i in range(1,NWAT):
        soln += specin[0].header['WAT2_0%02d'%(i)]
        if(len(specin[0].header['WAT2_0%02d'%(i)]) == 67):
            soln += ' '
    wsolarr = soln.split('"')
    # setup variables
    apStart = np.zeros(NAPS)
    apStep = np.zeros(NAPS*2+1)
    WAVE = np.zeros((NAPS,NPIX))
    # get data
    FLUX = specin[0].data[0]
    SIGMA = specin[0].data[2]
    # fill in wavelength values
    iapp = 0
    for i in range(1,NAPS*2+1,2):
        apStart[iapp] = wsolarr[i].split()[3]
        apStep[iapp] = wsolarr[i].split()[4]
        for j in range(NPIX):
            WAVE[iapp][j] = apStart[iapp] + j*apStep[iapp]
        iapp += 1
    specin.close()
    return (NAME,HJD,VHELIO,WAVE,FLUX,SIGMA)

# fit continuum using specutils functions
def contiFit(APER,LOW,HIGH,WAVE,FLUX,SIGMA):
    # fit continuum in window +/- 5 angstroms
    mask = (WAVE[APER] > LOW-5.) & (WAVE[APER] < HIGH+5.)
    bkgrfit = getBkgr(WAVE[APER][mask],FLUX[APER][mask],SIGMA[APER][mask])
    mask = (WAVE[APER] > LOW) & (WAVE[APER] < HIGH)
    CONTWAVE = WAVE[APER][mask]
    ycont = bkgrfit(CONTWAVE*u.AA)
    CONTFLUX = np.zeros(len(CONTWAVE))
    CONTSIGMA = np.zeros(len(CONTWAVE))
    CONTFLUX = FLUX[APER][mask]/ycont
    CONTSIGMA = SIGMA[APER][mask]/FLUX[APER][mask]*CONTFLUX
    return (CONTWAVE,CONTFLUX,CONTSIGMA)

# fit the line with astropy and find the centroid & equivalent width using specutils
def fitValues(APER,LWIN,HWIN,MEAN,DEPTH,STDDEV,CONTI,LOW,HIGH,WAVE,FLUX,SIGMA):
    CONTWAVE,CONTFLUX,CONTSIGMA = contiFit(APER,LOW,HIGH,WAVE,FLUX,SIGMA)

    # fit line with a Gaussian profile
    try:
        fmean, fmeanErr, fdepth, fdepthErr, fwidth, fwidthErr, fconti, fcontiErr, residual = fitLine(CONTI,DEPTH,MEAN,STDDEV,CONTWAVE,CONTFLUX,CONTSIGMA)
    except:
        fmean = fmeanErr = fdepth = fdepthErr = fwidth = fwidthErr = fconti = fcontiErr = residual = np.nan

    # get width for future measurements
    loLine = fmean - 3.0*fwidth
    if(loLine < LWIN):
        loLine = LWIN

    hiLine = fmean + 3.0*fwidth
    if(hiLine > HWIN):
        hiLine = HWIN

    if((hiLine - loLine) < 0.2):
        hiLine = HWIN
        loLine = LWIN

    bottom = fconti + 1.2*fdepth
    if(bottom > 0.8*min(CONTFLUX)):
        bottom = 0.8*min(CONTFLUX)

    # calculate the equivalent width from the continuum fit
    sumeqw = 0.0
    for i in range(len(CONTWAVE)):
        #if( (CONTWAVE[i] > LWIN) & (CONTWAVE[i] < HWIN) ):
        if( (CONTWAVE[i] > loLine) & (CONTWAVE[i] < hiLine) ):
            sumeqw += (fconti-CONTFLUX[i])/fconti*(CONTWAVE[i]-CONTWAVE[i-1])
    # calculate equivalent width from Gaussian integral
    integral = -1.0 / fconti * fdepth * fwidth * np.sqrt(2.0*np.pi)
    errorew = integral * np.sqrt( np.power(fdepthErr/fdepth,2.0) +
                                      np.power(fwidthErr/fwidth,2.0) )

    halfmax = fwhm = emax = fwem = 0.
    ## find centroid and equivalent width in window
    try:
        center,eqw,halfmax,fwhm,lhm,rhm,emax,fwem,lem,rem,lskew,lskewErr,lkurt,lkurtErr = lineMeasure(CONTWAVE,CONTFLUX,CONTSIGMA,loLine,hiLine,fconti,bottom)
    except:
        center = eqw = halfmax = fwhm = lhm = rhm = emax = fwem = lem = rem = lskew = lskewErr = lkurt = lkurtErr = np.nan

    return (fmean,fmeanErr,fdepth,fdepthErr,fwidth,fwidthErr,fconti,fcontiErr,residual,sumeqw,integral,errorew,center,eqw,halfmax,fwhm,lhm,rhm,emax,fwem,lem,rem,lskew,lskewErr,lkurt,lkurtErr)

# put output in a file
def saveLine(fout,fname,name,hjd,fmean,fmeanErr,fdepth,fdepthErr,fwidth,fwidthErr,fconti,fcontiErr,residual,sumeqw,integral,errorew,center,eqw,halfmax,fwhm,lhm,rhm,emax,fwem,lem,rem,lskew,lskewErr,lkurt,lkurtErr):
    with open(fout,'a') as f:
        f.write("%s %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n"%
                        (fname,name,hjd,fmean,fmeanErr,fdepth,fdepthErr,fwidth,fwidthErr,fconti,fcontiErr,residual,sumeqw,integral,errorew,center,eqw,halfmax,fwhm,lhm,rhm,emax,fwem,lem,rem,lskew,lskewErr,lkurt,lkurtErr))

#### Create the output file
with open('auto.txt','w') as f:
    f.write("file name HJD fitMean fitMeanErr fitDepth fitDepthErr fitWidth fitWidthErr fitCont fitContErr fitResidual fitEQW fitIntegral fitIntegralErr centroid EQW halfmax FWHM LHM RHM EMAX FWEM LEM REM skewness skewErr kurtosis kurtErr\n")

#### Read the input file
with open("autoinput.dat","r") as f:
    fullinput = f.readlines()
    eachline = [x.strip() for x in fullinput]

#### Loop over all the lines
for i in range(1,len(eachline)):
    thisline = eachline[i].split()

    froot = thisline[0]
    vlos = thisline[1]
    elem = thisline[2]
    lzero = thisline[3]
    aperture = int(thisline[4])
    linecen = float(thisline[5])
    lowbound = float(thisline[6])
    highbound = float(thisline[7])
    
    niteN = froot[1]

    try:
        name,hjd,vhelio,wave,flux,sigma = retrieveInput("/Users/kcarrell/MyGoogleDrive/Research/Students/LineMeasurements/nite%s/%s.ec.fits"%(niteN,froot))
    except:
        saveLine('auto.txt',froot,"BAD",-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.,-9999.)
        continue

    fmean,fmeanErr,fdepth,fdepthErr,fwidth,fwidthErr,fconti,fcontiErr,residual,sumeqw,integral,errorew,center,eqw,halfmax,fwhm,lhm,rhm,emax,fwem,lem,rem,lskew,lskewErr,lkurt,lkurtErr = fitValues(aperture-1,lowbound,highbound,linecen,-0.2,0.15,1.0,lowbound,highbound,wave,flux,sigma)
    #fitValues(48,4128.975,4130.975,4129.975,-0.2,0.15,1.0,4128.975,4130.975,wave,flux,sigma)
    #fitValues(48,4129.55,4130.5,4130.0,-0.2,0.12,1.0,4128.0,4132.0,wave,flux,sigma)

    # model_line = models.Const1D(fconti) + models.Gaussian1D(amplitude=fdepth, mean=fmean, stddev=fwidth)
    # contwave,conflux,contsigma = contiFit(aperture-1,lowbound,highbound,wave,flux,sigma)
    # plt.plot(contwave,conflux)
    # plt.plot(contwave,model_line(contwave),linewidth=2)
    # plt.axvline(x=center,linewidth=4,color='black')
    # plt.axvline(x=fmean-3.0*fwidth,linewidth=4,color='gray',alpha=0.5)
    # plt.axvline(x=fmean+3.0*fwidth,linewidth=4,color='gray',alpha=0.5)
    # plt.show()

    saveLine('auto.txt',froot,name,hjd,fmean,fmeanErr,fdepth,fdepthErr,fwidth,fwidthErr,fconti,fcontiErr,residual,sumeqw,integral,errorew,center,eqw,halfmax,fwhm,lhm,rhm,emax,fwem,lem,rem,lskew,lskewErr,lkurt,lkurtErr)

    print(i,froot,linecen,fmean,center)
