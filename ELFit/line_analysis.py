from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.analysis import centroid, equivalent_width
from astropy.modeling import models, fitting
from astropy.nddata import VarianceUncertainty
from astropy import units as u
from scipy.stats import skew, kurtosis
from random import uniform
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyWarning

# fit continuum using specutils functions
def getBkgr(wave,flux,sigma):
    spec = Spectrum1D(flux=flux*u.dimensionless_unscaled, spectral_axis=wave*u.AA, uncertainty=VarianceUncertainty(sigma*u.dimensionless_unscaled))
    return fit_generic_continuum(spec)


def lineMeasure(wave,flux,sigma,LWIN,HWIN,CONT,BOT):
    # find centroid and equivalent width in window
    spec = Spectrum1D(flux=flux*u.dimensionless_unscaled, spectral_axis=wave*u.AA, uncertainty=VarianceUncertainty(sigma*u.dimensionless_unscaled))
    specregion = SpectralRegion(LWIN*u.AA, HWIN*u.AA)
    center = centroid(spec, specregion)/u.AA
    eqwidth = equivalent_width(spec, regions=specregion)/u.AA
    peak = np.interp(center,wave,flux)
    halfmax = 1.0 - (1.0 - peak)/2.0
    #emax = 1.0 - (1.0 - peak)/np.e
    emax = 1.0 - (1.0 - peak)/10.0
    lft_msk = (wave<center) & (wave>LWIN)
    rght_msk = (wave>center) & (wave<HWIN)
    left = np.interp(halfmax,np.flip(flux[lft_msk]),np.flip(wave[lft_msk]))
    right = np.interp(halfmax,flux[rght_msk],wave[rght_msk])
    fwhm = right - left
    elft = np.interp(emax,np.flip(flux[lft_msk]),np.flip(wave[lft_msk]))
    erght = np.interp(emax,flux[rght_msk],wave[rght_msk])
    fwem = erght - elft
    vskew = []
    vkurt = []
    for i in range(5):
        xran = []
        while(len(xran)<10000):
            testx = uniform(LWIN,HWIN)
            testy = uniform(BOT,CONT)
            if(testy > np.interp(testx,wave,flux)):
                xran.append(testx)
        vskew.append(skew(xran))
        vkurt.append(kurtosis(xran))
    return center,eqwidth,halfmax,fwhm,(center-left)/fwhm,(right-center)/fwhm,emax,fwem,(center-elft)/fwem,(erght-center)/fwem,np.mean(vskew),np.std(vskew),np.mean(vkurt),np.std(vkurt)



def fitLine(gcon, gdep, gmean, gstd, wave, flux, sigma):
    model_line = models.Const1D(gcon) + models.Gaussian1D(amplitude=gdep, mean=gmean, stddev=gstd)
    warnings.simplefilter('ignore', category=AstropyWarning)
    fitter_line = fitting.LevMarLSQFitter()
    bestfit_line = fitter_line(model_line,wave,flux,weights=1./sigma)
    #with warnings.catch_warnings():
    #    warnings.simplefilter('ignore', category=AstropyWarning)
    #    bestfit_line = fitter_line(model_line,wave,flux,weights=1./sigma)
    try:
        cov_diag = np.diag(fitter_line.fit_info['param_cov'])
    except:
        cov_diag = np.empty(4)*np.nan

    fmean     = bestfit_line.mean_1.value
    fmeanErr  = np.sqrt(cov_diag[2])
    fdepth    = bestfit_line.amplitude_1.value
    fdepthErr = np.sqrt(cov_diag[1])
    fwidth    = bestfit_line.stddev_1.value
    fwidthErr = np.sqrt(cov_diag[3])
    fconti    = bestfit_line.amplitude_0.value
    fcontiErr = np.sqrt(cov_diag[0])

    residual = np.sum( sigma*(flux - bestfit_line(wave))**2 )

    return fmean, fmeanErr, fdepth, fdepthErr, fwidth, fwidthErr, fconti, fcontiErr, residual
