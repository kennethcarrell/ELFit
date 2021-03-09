import warnings
from astropy.utils.exceptions import AstropyWarning
import os
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.nddata import VarianceUncertainty
from specutils.analysis import centroid, equivalent_width
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy import units as u
import tkinter
import tkinter.ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#warnings.simplefilter('ignore', UserWarning)
#warnings.simplefilter('ignore', category=AstropyWarning)

## Values for the 107" Coude
_NUMAPS_ = 54
_NPIXELS_ = 2048

class LineFitGUI:
    # initialize GUI window
    def __init__(self, master):
        self.FOUT = open("lines.dat","a+")
        self.FILENAME = str(' ')
        self.NAME = str(' ')
        self.HJD = []
        self.VHELIO = []
        self.residual = 0.
        self.nsteps = 0
        self.apStart = np.zeros(_NUMAPS_)
        self.apStep = np.zeros(_NUMAPS_*2+1)
        self.WAVE = np.zeros((_NUMAPS_,_NPIXELS_))
        self.FLUX = np.zeros((_NUMAPS_,_NPIXELS_))
        self.SIGMA = np.zeros((_NUMAPS_,_NPIXELS_))
        self.CONTWAVE = []
        self.CONTFLUX = []
        self.CONTSIGMA = []

        self.master = master
        self.master.title("Line Analysis GUI")

        self.plotFrame = tkinter.Frame(self.master)
        self.aperFrame = tkinter.Frame(self.master)

        self.fNameButton = tkinter.Button(self.plotFrame, text="Open File", fg="blue", command=self.retrieveInput)
        self.fNameButton.grid(row = 0, column = 0)

        self.sNAMELabel = tkinter.Label(self.plotFrame, text="Name: ")
        self.sNAMELabel.grid(row = 0, column = 1)

        self.saveButton = tkinter.Button(self.plotFrame, text="Save This Line", fg="green", command=self.saveLine)
        self.saveButton.grid(row = 0, column = 2)

        self.updateButton = tkinter.Button(self.plotFrame, text="Update Plot", fg="red", command=self.updatePlot)
        self.updateButton.grid(row = 0, column = 3)
        self.updateButton['state'] = 'disabled'

        self.fitButton = tkinter.Button(self.plotFrame, text="Fit Values", fg="red", command=self.fitValues)
        self.fitButton.grid(row = 0, column = 4)
        self.fitButton['state'] = 'disabled'

        tkinter.ttk.Separator(self.plotFrame, orient='horizontal').grid(row=1,column=0,columnspan=5,sticky='ew')

        self.appLabel = tkinter.Label(self.plotFrame, text="Aperture")
        self.appLabel.grid(row = 2, column = 4)
        #self.apertures = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
        self.apertures = [*range(_NUMAPS_)]
        self.apVal = tkinter.IntVar(self.plotFrame)
        self.apVal.set(self.apertures[0])
        self.aperture = tkinter.OptionMenu(self.plotFrame,self.apVal,*self.apertures)
        self.aperture.grid(row = 3, column = 4)

        self.lowwindowLabel = tkinter.Label(self.plotFrame, text="Low Window:")
        self.lowwindowLabel.grid(row = 3, column = 0)
        self.lowwindow = tkinter.Entry(self.plotFrame, width=10)
        self.lowwindow.grid(row = 3, column = 1)

        self.highwindowLabel = tkinter.Label(self.plotFrame, text="High Window:")
        self.highwindowLabel.grid(row = 3, column = 2)
        self.highwindow = tkinter.Entry(self.plotFrame, width=10)
        self.highwindow.grid(row = 3, column = 3)

        tkinter.ttk.Separator(self.plotFrame, orient='horizontal').grid(row=4,column=0,columnspan=5,sticky='ew')

        self.contiLabel = tkinter.Label(self.plotFrame, text="Continuum")
        self.contiLabel.grid(row = 5, column = 0)
        self.continuum = tkinter.Entry(self.plotFrame,width=10)
        self.continuum.grid(row = 6, column = 0)
        self.contErr = tkinter.Entry(self.plotFrame,width=10)
        self.contErr.grid(row = 7, column = 0)

        self.meanLabel = tkinter.Label(self.plotFrame, text="Mean")
        self.meanLabel.grid(row = 5, column = 1)
        self.fitMean = tkinter.Entry(self.plotFrame,width=10)
        self.fitMean.grid(row = 6, column = 1)
        self.meanErr = tkinter.Entry(self.plotFrame,width=10)
        self.meanErr.grid(row = 7, column = 1)

        self.depthLabel = tkinter.Label(self.plotFrame, text="Depth")
        self.depthLabel.grid(row = 5, column = 2)
        self.fitDepth = tkinter.Entry(self.plotFrame,width=10)
        self.fitDepth.grid(row = 6, column = 2)
        self.depthErr = tkinter.Entry(self.plotFrame,width=10)
        self.depthErr.grid(row = 7, column = 2)

        self.widthLabel = tkinter.Label(self.plotFrame, text="Width")
        self.widthLabel.grid(row = 5, column = 3)
        self.fitWidth = tkinter.Entry(self.plotFrame,width=10)
        self.fitWidth.grid(row = 6, column = 3)
        self.widthErr = tkinter.Entry(self.plotFrame,width=10)
        self.widthErr.grid(row = 7, column = 3)

        self.calceqwLabel = tkinter.Label(self.plotFrame, text="Calc EQW")
        self.calceqwLabel.grid(row = 5, column = 4)
        self.calceqw = tkinter.Entry(self.plotFrame,width=10)
        self.calceqw.grid(row = 6, column = 4)

        tkinter.ttk.Separator(self.plotFrame, orient='horizontal').grid(row=8,column=0,columnspan=5,sticky='ew')

        self.centroidLabel = tkinter.Label(self.plotFrame, text="Centroid")
        self.centroidLabel.grid(row = 9, column = 0)
        self.centroid = tkinter.Entry(self.plotFrame,width=10)
        self.centroid.grid(row = 10, column = 0)

        self.eqwLabel = tkinter.Label(self.plotFrame, text="Eq Width")
        self.eqwLabel.grid(row = 9, column = 1)
        self.eqwidth = tkinter.Entry(self.plotFrame,width=10)
        self.eqwidth.grid(row = 10, column = 1)

        self.inteqwLabel = tkinter.Label(self.plotFrame, text="Integral Width")
        self.inteqwLabel.grid(row = 9, column = 3)
        self.inteqw = tkinter.Entry(self.plotFrame,width=10)
        self.inteqw.grid(row = 10, column = 3)
        self.inteqwErr = tkinter.Entry(self.plotFrame,width=10)
        self.inteqwErr.grid(row = 10, column = 4)

        self.figLine, self.figLAx = plt.subplots(figsize=(8,6))
        self.canLine = FigureCanvasTkAgg(self.figLine, self.plotFrame)
        self.canLine.get_tk_widget().grid(row = 11, column = 0, columnspan=5)

        self.lowRangeLabel = tkinter.Label(self.plotFrame, text="Plot Left:")
        self.lowRangeLabel.grid(row = 12, column = 0)
        self.lowRange = tkinter.Entry(self.plotFrame,width=10)
        self.lowRange.grid(row = 12, column = 1)

        self.highRangeLabel = tkinter.Label(self.plotFrame, text="Plot Right:")
        self.highRangeLabel.grid(row = 12, column = 2)
        self.highRange = tkinter.Entry(self.plotFrame,width=10)
        self.highRange.grid(row = 12, column = 3)

        self.plotMinLabel = tkinter.Label(self.plotFrame, text="Plot Min:")
        self.plotMinLabel.grid(row = 13, column = 0)
        self.plotMin = tkinter.Entry(self.plotFrame,width=10)
        self.plotMin.grid(row = 13, column = 1)

        self.plotMaxLabel = tkinter.Label(self.plotFrame, text="Plot Max:")
        self.plotMaxLabel.grid(row = 13, column = 2)
        self.plotMax = tkinter.Entry(self.plotFrame,width=10)
        self.plotMax.grid(row = 13, column = 3)

        self.residualLabel = tkinter.Label(self.plotFrame, text="Residual:")
        self.residualLabel.grid(row = 12, column = 4)

        self.savePlotButton = tkinter.Button(self.plotFrame, text="Save Plot", fg="green", command=self.savePlot)
        self.savePlotButton.grid(row = 13, column = 4)

        self.plotFrame.pack(side="left")
        self.aperlistbox = tkinter.Listbox(self.master,fg="gray",font="TkFixedFont",width=30)
        self.aperlistbox.pack(side="left",fill="y")
        #self.aperlistbox.configure(state=tkinter.DISABLE)


    ## Open an input file
    def retrieveInput(self):
        # open a dialog to get file
        fname = tkinter.filedialog.askopenfilename()
        specin = fits.open(fname)
        endParse = fname.split("/")
        self.FILENAME = endParse[len(endParse)-1].split(".")[0]
        # get info from header
        self.NAME = specin[0].header['OBJECT']
        self.sNAMELabel['text'] = "Name: %s\n%s"%(self.NAME,self.FILENAME)
        self.HJD = specin[0].header['HJD']
        self.VHELIO = specin[0].header['VHELIO']
        # get the wavelength solution
        soln = ''
        for i in range(1,61):
            soln += specin[0].header['WAT2_0%02d'%(i)]
            if(len(specin[0].header['WAT2_0%02d'%(i)]) == 67):
                soln += ' '
        wsolarr = soln.split('"')
        # get data
        self.FLUX = specin[0].data[0]
        self.SIGMA = specin[0].data[2]
        # fill in wavelength values
        iapp = 0
        for i in range(1,109,2):
            self.apStart[iapp] = wsolarr[i].split()[3]
            self.apStep[iapp] = wsolarr[i].split()[4]
            for j in range(_NPIXELS_):
                self.WAVE[iapp][j] = self.apStart[iapp] + j*self.apStep[iapp]
            iapp += 1
        specin.close()
        # OK to plot now
        self.updateButton['state'] = 'normal'
        self.fitButton['state'] = 'normal'
        self.aperlistbox.delete(0,tkinter.END)
        for i in range(_NUMAPS_):
            self.aperlistbox.insert(tkinter.END,"%2d: %10.5f -> %10.5f"%(i+1,self.WAVE[i][0],self.WAVE[i][-1]))
        self.aperlistbox['state'] = 'disable'

    # fit continuum using specutils functions
    def contiFit(self):
        APER = self.apVal.get() - 1
        LOW = float(self.lowRange.get())
        HIGH = float(self.highRange.get())
        # fit continuum in window +/- 5 angstroms
        mask = (self.WAVE[APER] > LOW-5.) & (self.WAVE[APER] < HIGH+5.)
        spec = Spectrum1D(flux=self.FLUX[APER][mask]*u.dimensionless_unscaled, spectral_axis=self.WAVE[APER][mask]*u.AA, uncertainty=VarianceUncertainty(self.SIGMA[APER][mask]*u.dimensionless_unscaled))
        bkgrfit = fit_generic_continuum(spec)
        mask = (self.WAVE[APER] > LOW) & (self.WAVE[APER] < HIGH)
        self.CONTWAVE = self.WAVE[APER][mask]
        ycont = bkgrfit(self.CONTWAVE*u.AA)
        self.CONTFLUX = np.zeros(len(self.CONTWAVE))
        self.CONTSIGMA = np.zeros(len(self.CONTWAVE))
        self.CONTFLUX = self.FLUX[APER][mask]/ycont
        self.CONTSIGMA = self.SIGMA[APER][mask]/self.FLUX[APER][mask]*self.CONTFLUX

    # fit the line with astropy and find the centroid & equivalent width using specutils
    def fitValues(self):
        # get values from boxes
        APER = self.apVal.get() - 1
        try:
            MEAN = float(self.fitMean.get())
            DEPTH = float(self.fitDepth.get())
            STDDEV = float(self.fitWidth.get())
            CONTI = float(self.continuum.get())
            LOW = float(self.lowRange.get())
            HIGH = float(self.highRange.get())
        except:
            return
        try:
            LWIN = float(self.lowwindow.get())
            HWIN = float(self.highwindow.get())
        except:
            LWIN = float(self.lowRange.get())
            HWIN = float(self.highRange.get())
        self.contiFit()
        # find centroid and equivalent width in window
        spec = Spectrum1D(flux=self.CONTFLUX*u.dimensionless_unscaled, spectral_axis=self.CONTWAVE*u.AA, uncertainty=VarianceUncertainty(self.CONTSIGMA*u.dimensionless_unscaled))
        self.centroid.delete(0, tkinter.END)
        self.centroid.insert(0, "%.4f"%(centroid(spec, SpectralRegion(LWIN*u.AA, HWIN*u.AA))/u.AA))
        self.eqwidth.delete(0, tkinter.END)
        self.eqwidth.insert(0, "%.6f"%(equivalent_width(spec, regions=SpectralRegion(LWIN*u.AA, HWIN*u.AA))/u.AA))

        # fit line with a Gaussian profile
        model_line = models.Const1D(CONTI) + models.Gaussian1D(amplitude=DEPTH, mean=MEAN, stddev=STDDEV)
        fitter_line = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bestfit_line = fitter_line(model_line,self.CONTWAVE,self.CONTFLUX,weights=1./self.CONTSIGMA)
        try:
            cov_diag = np.diag(fitter_line.fit_info['param_cov'])
        except:
            cov_diag = np.empty(4)*np.nan
        # update values in boxes
        self.fitMean.delete(0, tkinter.END)
        self.fitMean.insert(0, "%.4f"%(bestfit_line.mean_1.value))
        self.meanErr.delete(0, tkinter.END)
        self.meanErr.insert(0, "%.4f"%(np.sqrt(cov_diag[2])))
        self.fitDepth.delete(0, tkinter.END)
        self.fitDepth.insert(0, "%.6f"%(bestfit_line.amplitude_1.value))
        self.depthErr.delete(0, tkinter.END)
        self.depthErr.insert(0, "%.6f"%(np.sqrt(cov_diag[1])))
        self.fitWidth.delete(0, tkinter.END)
        self.fitWidth.insert(0, "%.6f"%(bestfit_line.stddev_1.value))
        self.widthErr.delete(0, tkinter.END)
        self.widthErr.insert(0, "%.6f"%(np.sqrt(cov_diag[3])))
        self.continuum.delete(0, tkinter.END)
        self.continuum.insert(0, "%.6f"%(bestfit_line.amplitude_0.value))
        self.contErr.delete(0, tkinter.END)
        self.contErr.insert(0, "%.6f"%(np.sqrt(cov_diag[0])))
        # calculate the equivalent width from the continuum fit
        sumeqw = 0.0
        conti = bestfit_line.amplitude_0.value
        for i in range(len(self.CONTWAVE)):
            if( (self.CONTWAVE[i] > LWIN) & (self.CONTWAVE[i] < HWIN) ):
                sumeqw += (conti-self.CONTFLUX[i])/conti*(self.CONTWAVE[i]-self.CONTWAVE[i-1])
        self.calceqw.delete(0, tkinter.END)
        self.calceqw.insert(0, "%.6f"%(sumeqw))
        # calculate equivalent width from Gaussian integral
        integral = -1.0 / float(self.continuum.get()) * float(self.fitDepth.get()) * float(self.fitWidth.get()) * np.sqrt(2.0*np.pi)
        self.inteqw.delete(0, tkinter.END)
        self.inteqw.insert(0, "%.6f"%(integral))
        errorew = integral * np.sqrt( np.power(float(self.depthErr.get())/float(self.fitDepth.get()),2.0) +
                                          np.power(float(self.widthErr.get())/float(self.fitWidth.get()),2.0) )
        self.inteqwErr.delete(0, tkinter.END)
        self.inteqwErr.insert(0, "%.6f"%(errorew))
        # calculate the residual of the fit
        self.residual = np.sum( self.CONTSIGMA*(self.CONTFLUX - bestfit_line(self.CONTWAVE))**2 )
        self.residualLabel['text'] = "Residual:\n%.6f"%(self.residual)
        self.updatePlot()

    # update the plot in the GUI
    def updatePlot(self):
        plt.close('all')
        self.canLine.get_tk_widget().grid_forget()
        self.figLine.clear()
        self.figLine, self.figLAx = plt.subplots(figsize=(8,6))
        # get info from boxes
        APER = self.apVal.get() - 1
        try:
            LOW = float(self.lowRange.get())
            HIGH = float(self.highRange.get())
        except:
            return
        try:
            MEAN = float(self.fitMean.get())
            DEPTH = float(self.fitDepth.get())
            STDDEV = float(self.fitWidth.get())
            CONTI = float(self.continuum.get())
        except:
            MEAN = (HIGH+LOW)/2.0
            DEPTH = 0.001
            STDDEV = 0.001
            CONTI = 1.0
        try:
            CENT = float(self.centroid.get())
        except:
            CENT = 0.0
        try:
            LWIN = float(self.lowwindow.get())
            HWIN = float(self.highwindow.get())
        except:
            LWIN = LOW
            HWIN = HIGH
        try:
            TOP = float(self.plotMax.get())
            BOTTOM = float(self.plotMin.get())
        except:
            TOP = 1.2
            BOTTOM = 0.0

        model_line = models.Const1D(CONTI) + models.Gaussian1D(amplitude=DEPTH, mean=MEAN, stddev=STDDEV)
        # fit continuum and plot data
        self.contiFit()
        self.figLAx.plot(self.CONTWAVE,self.CONTFLUX)
        # centroid line
        self.figLAx.axvline(x=CENT,linewidth=4,color='black')
        # window border
        self.figLAx.axvline(x=LWIN,linewidth=4,color='gray',alpha=0.5)
        self.figLAx.axvline(x=HWIN,linewidth=4,color='gray',alpha=0.5)
        # fitted line
        self.figLAx.plot(self.CONTWAVE,model_line(self.CONTWAVE),linewidth=2)
        self.figLAx.grid()
        self.figLAx.set_xlim(LOW,HIGH)
        self.figLAx.set_ylim(BOTTOM,TOP)
        self.canLine = FigureCanvasTkAgg(self.figLine, self.plotFrame)
        self.canLine.get_tk_widget().grid(row = 11, column = 0, columnspan=5)

    # put output in a file
    def saveLine(self):
        self.FOUT.write("%s %s %f %f %d %.4f %.6f %.6f %.6f %.6f %.4f %.6f %.6f %.6f %.6f %.6f %.6f\n"%
                            (self.FILENAME, self.NAME, self.HJD, self.VHELIO, np.int8(self.apVal.get()),
                            np.float32(self.centroid.get()), np.float32(self.eqwidth.get()),
                            np.float32(self.calceqw.get()),
                            np.float32(self.inteqw.get()), np.float32(self.inteqwErr.get()),
                            np.float32(self.fitMean.get()), np.float32(self.meanErr.get()),
                            np.float32(self.fitWidth.get()), np.float32(self.widthErr.get()),
                            np.float32(self.fitDepth.get()), np.float32(self.depthErr.get()),
                            self.residual))
        self.FOUT.flush()
        os.fsync(self.FOUT.fileno())

    def savePlot(self):
        fname = tkinter.filedialog.asksaveasfilename()
        if not fname:
            return
        plt.title("%s %s"%(self.FILENAME,self.NAME))
        self.figLine.savefig(fname)
        plt.title("")
