import numpy as np
from numpy.fft import rfft,rfftfreq,irfft

def OptimumFilterAmplitude(Signal, Template, NoisePSD, Fs, withDelay=True, normalize=False, coupling='AC'):
    """Function that applies an optimum filter to signal, given a noise PSD and a template. This function supports optimum filters with and without time delay.
    
    Args:
        Signal: The trace that will be filtered
        Template: The template of the signal that we are trying to find in the signal
        NoisePSD: Power spectral density for the noise in the signal
        withDelay: Boolean flag to choose whether or not to filter with a time delay
        normalize: Boolean flag that normalizes the template, if it has not yet been normalized
        coupling: Choose whether or not to use the DC value of the noise PSD. Usually, this value is inaccurate, so it can be ignored by setting coupling="AC", which sets the DC value to infinity. If the DC value should be used, set coupling = "DC"
        
    Returns:
        A: The optimal amplitude of the signal
        t0: The time delay of the signal
        Xr: The value of chi-squared at where the optimal value is
    
    """
    dt = 1.0/Fs
    Ns = float(len(Signal))
    T = Ns*dt
    dnu = 1.0/T

    if(normalize):
        tRange = max(Template) - min(Template)
        Template/=tRange

    #take one-sided fft of Signal and Template
    Sf = rfft(Signal)
    Tf = rfft(Template)

    #check for compatibility between PSD and fft
    if (len(NoisePSD) != len(Sf)):
        raise ValueError("PSD length incompatible with signal size")
    
    #take squared noise PSD
    J = NoisePSD**2.0
    
    #If AC coupled, the 0 component of the PSD is non-sensical
    #If DC coupled, ignoring the DC component will still give the correct amplitude
    if (coupling == 'AC'):
        J[0] = np.inf

    #find optimum filter and norm
    OF = Tf.conjugate()/J
    Norm = np.real(OF.dot(Tf))
    OFp = OF/Norm

    Sfilt = OFp*Sf

    #this factor is derived from the need to convert the dft to continuous units, and then get a reduced chi-square
    chiScale = 2*dt/(Ns**2)

    #compute OF with delay
    if (withDelay):
        #have to correct for np rfft convention by multiplying by N/2
        At = irfft(Sfilt)*Ns/2.0
        
        #signal pary of chi-square
        chi0 = np.real(np.dot(Sf.conjugate()/J,Sf))
        
        #fitting part of chi-square
        chit = (At**2)*Norm
        
        #sum parts of chi-square
        chi = (chi0 - chit)*chiScale
        
        #find time of best-fit
        bInd = np.argmin(chi)
        A = At[bInd]
        Xr = chi[bInd]
        t0 = bInd*dt

        if (t0 == T):
            t0 -= T

    #compute OF amplitude no delay
    else:
        A = np.real(np.sum(Sfilt))
        t0 = 0.0
    
        #signal pary of chi-square
        chi0 = np.real(np.dot(Sf.conjugate()/J,Sf))

        #fitting part of chi-square
        chit = (A**2)*Norm

        Xr = (chi0-chit)*chiScale

    return A, t0, Xr