import numpy as np
from numpy import pi
from scipy.optimize import least_squares,fsolve
from scipy.fftpack import fft, ifft, fftfreq
from scipy.stats import skew
import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
from General import autoCuts
from itertools import izip,repeat

def stdComplex(x,axis=0):
    """Function to return complex standard deviation (individually computed for real and imaginary components) for an array of complex values.
    
    Args:
        x: An array of complex values from which we want the complex standard deviation.
        axis: Which axis to take the standard deviation of (should be used if the dimension of the array is greater than 1)
        
    Returns:
        std_complex: The complex standard deviation of the inputted array, along the specified axis.
    """
    rstd=np.std(x.real,axis=axis)
    istd=np.std(x.imag,axis=axis)
    std_complex = rstd+1.0j*istd
    return std_complex

def slope(x,y,removeMeans=True):
    """Computes the maximum likelihood slope of a set of x and y points
    
    Args:
        x: Array of real-valued independent variables
        y: Array of real-valued dependent variables
        removeMeans: Boolean value on whether the mean needs to be removed from x and y. Set to false if mean has already been removed from x

    Returns:
        slope: Maximum likelihood slope estimate calculated as sum((y-<y>)*(x-<x>))/sum((x-<x>)^2)
    """
    
    if(removeMeans):
        xTemp = x - np.mean(x)
        yTemp = y - np.mean(y)
    else:
        xTemp = x
        yTemp = y

    return np.sum(xTemp*yTemp)/np.sum(xTemp**2)
    
def OnePoleImpedance(freq,A,tau2):
    """Function to calculate the impedance (dVdI) of a TES with the 1-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), superconducting: A=Rl, normal: A = Rl+Rn
        tau2: The fit parameter tau2 in the complex impedance (in s), superconducting: tau2=L/Rl, normal: tau2=L/(Rl+Rn)
        
    Returns:
        dVdI: The complex impedance of the TES with the 1-pole fit
    
    """
    
    dVdI=(A*(1.0+2.0j*pi*freq*tau2))
    return dVdI

def OnePoleAdmittance(freq,A,tau2):
    """Function to calculate the admittance (dIdV) of a TES with the 1-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), superconducting: A=Rl, normal: A = Rl+Rn
        tau2: The fit parameter tau2 in the complex impedance (in s), superconducting: tau2=L/Rl, normal: tau2=L/(Rl+Rn)
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 1-pole fit
    
    """
    
    dVdI=OnePoleImpedance(freq,A,tau2)
    return (1.0/dVdI)

def TwoPoleImpedance(freq,A,B,tau1,tau2):
    """Function to calculate the impedance (dVdI) of a TES with the 2-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), A = Rl + R0*(1+beta)
        B: The fit parameter B in the complex impedance (in Ohms), B = R0*l*(2+beta)/(1-l) (where l is Irwin's loop gain)
        tau1: The fit parameter tau1 in the complex impedance (in s), tau1=tau0/(1-l)
        tau2: The fit parameter tau2 in the complex impedance (in s), tau2=L/(Rl+R0*(1+beta))
        
    Returns:
        dVdI: The complex impedance of the TES with the 2-pole fit
    
    """
    
    dVdI=(A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1))
    return dVdI

def TwoPoleAdmittance(freq,A,B,tau1,tau2):
    """Function to calculate the admittance (dIdV) of a TES with the 2-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms), A = Rl + R0*(1+beta)
        B: The fit parameter B in the complex impedance (in Ohms), B = R0*l*(2+beta)/(1-l) (where l is Irwin's loop gain)
        tau1: The fit parameter tau1 in the complex impedance (in s), tau1=tau0/(1-l)
        tau2: The fit parameter tau2 in the complex impedance (in s), tau2=L/(Rl+R0*(1+beta))
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 2-pole fit
    
    """
    
    dVdI=TwoPoleImpedance(freq,A,B,tau1,tau2)
    return (1.0/dVdI)

def ThreePoleImpedance(freq,A,B,C,tau1,tau2,tau3):
    """Function to calculate the impedance (dVdI) of a TES with the 3-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms)
        B: The fit parameter B in the complex impedance (in Ohms)
        C: The fit parameter C in the complex impedance
        tau1: The fit parameter tau1 in the complex impedance (in s)
        tau2: The fit parameter tau2 in the complex impedance (in s)
        tau3: The fit parameter tau3 in the complex impedance (in s)
        
    Returns:
        dVdI: The complex impedance of the TES with the 3-pole fit
    
    """
    
    dVdI=(A*(1.0+2.0j*pi*freq*tau2))+(B/(1.0+2.0j*pi*freq*tau1-C/(1.0+2.0j*pi*freq*tau3)))
    return dVdI

def ThreePoleAdmittance(freq,A,B,C,tau1,tau2,tau3):
    """Function to calculate the admittance (dIdV) of a TES with the 3-pole fit
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        A: The fit parameter A in the complex impedance (in Ohms)
        B: The fit parameter B in the complex impedance (in Ohms)
        C: The fit parameter C in the complex impedance
        tau1: The fit parameter tau1 in the complex impedance (in s)
        tau2: The fit parameter tau2 in the complex impedance (in s)
        tau3: The fit parameter tau3 in the complex impedance (in s)
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 3-pole fit
    
    """
    
    dVdI=ThreePoleImpedance(freq,A,B,C,tau1,tau2,tau3)
    return (1.0/dVdI)

def TwoPoleImpedancePriors(freq,Rl,R0,beta,l,L,tau0):
    """Function to calculate the impedance (dVdI) of a TES with the 2-pole fit from Irwin's TES parameters
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        Rl: The load resistance of the TES (in Ohms)
        R0: The resistance of the TES (in Ohms)
        beta: The current sensitivity of the TES
        l: Irwin's loop gain
        L: The inductance in the TES circuit (in Henrys)
        tau0: The thermal time constant of the TES (in s)
        
    Returns:
        dVdI: The complex impedance of the TES with the 2-pole fit from Irwin's TES parameters
    
    """
    
    dVdI= Rl + R0*(1.0+beta) + 2.0j*pi*freq*L + R0 * l * (2.0+beta)/(1.0-l) * 1.0/(1.0+2.0j*freq*pi*tau0/(1.0-l))
    return dVdI

def TwoPoleAdmittancePriors(freq,Rl,R0,beta,l,L,tau0):
    """Function to calculate the admittance (dIdV) of a TES with the 2-pole fit from Irwin's TES parameters
    
    Args:
        freq: The frequencies for which to calculate the admittance (in Hz)
        Rl: The load resistance of the TES (in Ohms)
        R0: The resistance of the TES (in Ohms)
        beta: The current sensitivity of the TES, beta=d(log R)/d(log I)
        l: Irwin's loop gain, l = P0*alpha/(G*Tc)
        L: The inductance in the TES circuit (in Henrys)
        tau0: The thermal time constant of the TES (in s), tau0=C/G
        
    Returns:
        1.0/dVdI: The complex admittance of the TES with the 2-pole fit from Irwin's TES parameters
    
    """
    
    dVdI=TwoPoleImpedancePriors(freq,Rl,R0,beta,l,L,tau0)
    return (1.0/dVdI)


def YDI(x,A,B,C,tau1,tau2,tau3,sgAmp,Rsh,sgFreq,dutycycle):
    """Function to convert the fitted TES parameters for the complex impedance to a TES response to a square wave jitter in time domain.
    
    Args:
        x: Time values for the trace (in s)
        A: The fit parameter A in the complex impedance (in Ohms)
        B: The fit parameter B in the complex impedance (in Ohms)
        C: The fit parameter C in the complex impedance
        tau1: The fit parameter tau1 in the complex impedance (in s)
        tau2: The fit parameter tau2 in the complex impedance (in s)
        tau3: The fit parameter tau3 in the complex impedance (in s)
        sgAmp: The peak-to-peak size of the square wave jitter (in Amps)
        Rsh: The shunt resistance of the TES electronics (in Ohms)
        sgFreq: The frequency of the square wave jitter (in Hz)
        dutycycle: The duty cycle of the square wave jitter (between 0 and 1)
        
    Returns:
        np.real(St): The response of a TES to a square wave jitter in time domain with the given fit parameters. The real part is taken in order to ensure that the trace is real
    
    
    """
    
    tracelength=len(x)
    
    # get the frequencies for a DFT, based on the sample rate of the data
    dx=x[1]-x[0]
    freq=fftfreq(len(x),d=dx)
    
    # dIdV of fit in frequency space
    ci = ThreePoleAdmittance(freq,A,B,C,tau1,tau2,tau3)

    # analytic DFT of a duty cycled square wave
    Sf = np.zeros_like(freq)*0.0j
    
    # even frequencies are zero unless the duty cycle is not 0.5
    if (dutycycle==0.5):
        oddInds = ((np.abs(np.mod(np.absolute(freq/sgFreq),2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = 1.0j/(pi*freq[oddInds]/sgFreq)*sgAmp*Rsh*tracelength
    else:
        oddInds = ((np.abs(np.mod(np.abs(freq/sgFreq),2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = -1.0j/(2.0*pi*freq[oddInds]/sgFreq)*sgAmp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[oddInds]/sgFreq*dutycycle)-1)
        
        evenInds = ((np.abs(np.mod(np.abs(freq/sgFreq)+1,2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        evenInds[0] = False
        Sf[evenInds] = -1.0j/(2.0*pi*freq[evenInds]/sgFreq)*sgAmp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[evenInds]/sgFreq*dutycycle)-1)
    
    # convolve the square wave with the fit
    SfTES = Sf*ci
    
    # inverse FFT to convert to time domain
    St = ifft(SfTES)

    return np.real(St)

def SquareWaveGuessParams(trace,sgAmp,Rsh):
    """Function to guess the fit parameters for the 1-pole fit.
    
    Args:
        trace: The trace in time domain (in Amps).
        sgAmp: The peak-to-peak size of the square wave jitter (in Amps)
        Rsh: Shunt resistance of the TES electronics (in Ohms)
        
    Returns:
        A0: Guess of the fit parameter A (in Ohms)
        tau20: Guess of the fit parameter tau2 (in s)
    
    """
    
    dIs = max(trace) - min(trace)
    A0 = sgAmp*Rsh/dIs
    tau20 = 1.0e-6
    return A0,tau20

def GuessDIDVParams(trace,traceTopSlope,sgAmp,Rsh,L0=1.0e-7):
    """Function to find the fit parameters for either the 1-pole (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit. 
    
    Args:
        trace: The trace in time domain (in Amps)
        traceTopSlope: The flat parts of the trace (in Amps)
        sgAmp: The peak-to-peak size of the square wave jitter (in Amps)
        Rsh: Shunt resistance of the TES electronics (in Ohms)
        L0: The guess of the inductance (in Henries)
        
    Returns:
        A0: Guess of the fit parameter A (in Ohms)
        B0: Guess of the fit parameter B (in Ohms)
        tau10: Guess of the fit parameter tau1 (in s)
        tau20: Guess of the fit parameter tau2 (in s)
        isLoopGainSub1: Boolean flag that gives whether the loop gain is greater than one (False) or less than one (True)
        
    """
    
    # get the mean of the trace
    dIsmean = np.mean(trace)
    # mean of the top slope points
    dIsTopSlopemean = np.mean(traceTopSlope)
    #check if loop gain is less than or greater than one (check if we are inverted of not)
    isLoopGainSub1 = dIsTopSlopemean < dIsmean
    
    # the dIdV(0) can be estimate as twice the difference of the top slope points and the mean of the trace
    dIs0 = 2 * np.abs(dIsTopSlopemean-dIsmean)
    dIdV0 = dIs0/(sgAmp*Rsh)
    
    # beta can be estimated from the size of the overshoot
    # estimate size of overshoot as maximum of trace minus the dIsTopSlopemean
    dIsTop = np.max(trace)-dIsTopSlopemean
    dIsdVTop = dIsTop/(sgAmp*Rsh)
    A0 = 1.0/dIsdVTop
    tau20 = L0/A0
    
    if isLoopGainSub1:
        # loop gain < 1
        B0 = 1.0/dIdV0 - A0
        if B0 > 0.0:
            B0 = -B0 # this should be positive, but since the optimization algorithm checks both cases, we need to make sure it's negative, otherwise the guess will not be within the allowed bounds
        tau10 = -100e-6 # guess a slower tauI
    else:
        # loop gain > 1
        B0 = -1.0/dIdV0 - A0
        tau10 = -100e-7 # guess a faster tauI

    return A0,B0,tau10,tau20,isLoopGainSub1

def FitYFreq(freq,dIdV,yerr=None,A0=0.25,B0=-0.6,C0=-0.6,tau10=-1.0/(2*pi*5e2),tau20=1.0/(2*pi*1e5),tau30=0.0,dt=-10.0e-6,poles=2):
    """Function to find the fit parameters for either the 1-pole (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit. 
    
    Args:
        freq: Frequencies corresponding to the dIdV
        dIdV: Complex impedance extracted from the trace in frequency space
        yerr: Error at each frequency of the dIdV. Should be a complex number, e.g. yerr = yerr_real + 1.0j * yerr_imag, where yerr_real is the standard deviation of the real part of the dIdV, and yerr_imag is the standard deviation of the imaginary part of the dIdV
        A0: Guess of the fit parameter A (in Ohms)
        B0: Guess of the fit parameter B (in Ohms)
        C0: Guess of the fit parameter C
        tau10: Guess of the fit parameter tau1 (in s)
        tau20: Guess of the fit parameter tau2 (in s)
        tau30: Guess of the fit parameter tau3 (in s)
        dt: Guess of the time shift (in s)
        poles: The number of poles to use in the fit (should be 1, 2, or 3)
        
    Returns:
        popt: The fitted parameters for the specificed number of poles
        pcov: The corresponding covariance matrix for the fitted parameters
        cost: The cost of the the fit
        
    """
    
    if(poles==1):
        # assume the square wave is not inverted
        p0=(A0,tau20,dt)
        bounds1=((0.0,0.0,-1.0e-3),(np.inf,np.inf,1.0e-3))
        # assume the square wave is inverted
        p02=(-A0,tau20,dt)
        bounds2=((-np.inf,0.0,-1.0e-3),(0.0,np.inf,1.0e-3))
    elif(poles==2):
        # assume loop gain > 1, where B<0 and tauI<0
        p0=(A0,B0,tau10,tau20,dt)
        bounds1=((0.0,-np.inf,-np.inf,0.0,-1.0e-3),(np.inf,0.0,0.0,np.inf,1.0e-3))
        # assume loop gain < 1, where B>0 and tauI>0
        p02=(A0,-B0,-tau10,tau20,dt)
        bounds2=((0.0,0.0,0.0,0.0,-1.0e-3),(np.inf,np.inf,np.inf,np.inf,1.0e-3))
    elif(poles==3):
        # assume loop gain > 1, where B<0 and tauI<0
        p0=(A0,B0,C0,tau10,tau20,tau30,dt)
        bounds1=((0.0,-np.inf,-np.inf,-np.inf,0.0,0.0,-1.0e-3),(np.inf,0.0,0.0,0.0,np.inf,np.inf,1.0e-3))
        # assume loop gain < 1, where B>0 and tauI>0
        p02=(A0,-B0,-C0,-tau10,tau20,tau30,dt)
        bounds2=((0.0,0.0,0.0,0.0,0.0,0.0,-1.0e-3),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1.0e-3))
        
    def residual(params):
        # define a residual for the nonlinear least squares algorithm 
        fZ=freq
        # different functions for different amounts of poles
        if(poles==1):
            A,tau2,dt=params
            ci=OnePoleAdmittance(fZ,A,tau2) * np.exp(-2.0j*pi*fZ*dt)
        elif(poles==2):
            A,B,tau1,tau2,dt=params
            ci=TwoPoleAdmittance(fZ,A,B,tau1,tau2) * np.exp(-2.0j*pi*fZ*dt)
        elif(poles==3):
            A,B,C,tau1,tau2,tau3,dt=params
            ci=ThreePoleAdmittance(fZ,A,B,C,tau1,tau2,tau3) * np.exp(-2.0j*pi*fZ*dt)
        
        # the difference between the data and the fit
        diff=dIdV-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if(yerr is None):
            weights=1.0+1.0j
        else:
            weights=1.0/yerr.real+1.0j/yerr.imag
        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(fZ.size*2, dtype = np.float64)
        z1d[0:z1d.size:2] = diff.real*weights.real
        z1d[1:z1d.size:2] = diff.imag*weights.imag
        return z1d

    # res1 assumes loop gain > 1, where B<0 and tauI<0
    res1=least_squares(residual, p0, bounds=bounds1, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    # res2 assumes loop gain < 1, where B>0 and tauI>0
    res2=least_squares(residual, p02, bounds=bounds2, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    
    # check which loop gain casez gave the better fit
    if(res1['cost'] < res2['cost']):
        res=res1
    else:
        res=res2
        
    popt=res['x']
    cost=res['cost']
        
    # check if the fit failed (usually only happens when we reach maximum evaluations, likely when fitting assuming the wrong loop gain)
    if(not res1['success'] and not res2['success']):
        print("Fit failed: "+str(res1['status'])+", "+str(poles)+"-pole Fit")
        
    # take matrix product of transpose of jac and jac, take the inverse to get the analytic covariance matrix
    pcovinv = np.dot(res["jac"].transpose(),res["jac"])
    pcov = np.linalg.inv(pcovinv)
    
    return popt,pcov,cost

def ConvertToTESValues(popt,pcov,R0,Rl,dR0=0.001,dRl=0.001,poles=2):
    """Function to convert the fit parameters for either 1-pole (A, tau2, dt), 2-pole (A, B, tau1, tau2, dt), or 3-pole (A, B, C, tau1, tau2, tau3, dt) fit to the corresponding TES parameters: 1-pole (Rtot, L, R0, Rl, dt), 2-pole (Rl, R0, beta, l, L, tau0, dt), and 3-pole (no conversion done).
    
    Args:
        popt: The fit parameters for either the 1-pole, 2-pole, or 3-pole fit
        pcov: The corresponding covariance matrix for the fit parameters
        R0: The resistance of the TES (in Ohms)
        Rl: The load resistance of the TES circuit (in Ohms)
        dR0: The error in the R0 value (in Ohms)
        dRl: The error in the Rl value (in Ohms)
        poles: The fit that we will convert the parameters to (I should change this to be automatically known from length of popt)
        
    Returns:
        popt_out: The TES parameters for the specified fit
        pcov_out: The corresponding covariance matrix for the TES parameters
        
    """
    
    if(poles==1):
        ## one pole
        # extract fit parameters
        A  = popt[0]
        tau2 = popt[1]
        dt = popt[2]
        
        # convert fit parameters to Rtot=R0+Rl and L
        Rtot = A
        L = A*tau2
        
        popt_out = [Rtot,L,R0,Rl,dt]
        
        # create new covariance matrix (needs to be the correct size)
        pcov_orig = pcov
        pcov_in = np.zeros((5,5))
        row,col = np.indices((2,2))
        
        # populate the new covariance matrix with the uncertainties in R0, Rl, and dt
        pcov_in[row,col] = pcov_orig[row,col]
        vardt = pcov_orig[2,2]
        pcov_in[2,2] = dR0**2
        pcov_in[3,3] = dRl**2
        pcov_in[4,4] = vardt

        # calculate the Jacobian
        jac = np.zeros((5,5))
        jac[0,0] = 1             # dRtotdA
        jac[1,0] = tau2          # dLdA
        jac[1,1] = A             # dLdtau2
        jac[2,2] = 1             # dR0dR0
        jac[3,3] = 1             # dRldRl
        jac[4,4] = 1             # ddtddt
        
        # use the Jacobian to populate the rest of the covariance matrix
        jact = np.transpose(jac)
        pcov_out = np.dot(jac,np.dot(pcov_in,jact))
        
    elif (poles==2):
        ## two poles
        # extract fit parameters
        A = popt[0]
        B = popt[1]
        tau1 = popt[2]
        tau2 = popt[3]
        dt = popt[4]
        # get covariance matrix for beta, l, L, tau, R0, Rl, dt
        # create new covariance matrix (needs to be the correct size)
        pcov_orig = np.copy(pcov)
        pcov_in = np.zeros((7,7))
        row,col = np.indices((4,4))

        # populate the new covariance matrix with the uncertainties in R0, Rl, and dt
        pcov_in[row,col] = np.copy(pcov_orig[row,col])
        vardt = pcov_orig[4,4]
        pcov_in[4,4] = dRl**2
        pcov_in[5,5] = dR0**2
        pcov_in[6,6] = vardt
        
        # convert A, B tau1, tau2 to beta, l, L, tau
        beta  = (A-Rl)/R0 - 1.0
        l = B/(A+B+R0-Rl)
        L = A*tau2
        tau = tau1 * (A+R0-Rl)/(A+B+R0-Rl)
        popt_out = [Rl,R0,beta,l,L,tau,dt]
        
        # calculate the Jacobian
        jac = np.zeros((7,7))
        jac[0,4] = 1.0                              #dRldRl
        jac[1,5] = 1.0                              #dR0dR0
        jac[2,0] = 1.0/R0                           #dbetadA
        jac[2,4] = -1.0/R0                          #dbetadRl
        jac[2,5] = -(A-Rl)/R0**2.0                  #dbetadR0
        jac[3,0] = -B/(A+B+R0-Rl)**2.0              #dldA (l = Irwin's loop gain = (P0 alpha)/(G T0))
        jac[3,1] = (A+R0-Rl)/(A+B+R0-Rl)**2.0       #dldB
        jac[3,4] = B/(A+B+R0-Rl)**2.0               #dldRl
        jac[3,5] = -B/(A+B+R0-Rl)**2.0              #dldR0
        jac[4,0] = tau2                             #dLdA
        jac[4,3] = A                                #dLdtau2
        jac[5,0] = (tau1*B)/(A+B+R0-Rl)**2.0        #dtaudA
        jac[5,1] = -tau1*(A+R0-Rl)/(A+B+R0-Rl)**2.0 #dtaudB
        jac[5,2] = (A+R0-Rl)/(A+B+R0-Rl)            #dtaudtau1
        jac[5,4] = -B*tau1/(A+B+R0-Rl)**2.0         #dtaudRl
        jac[5,5] = B*tau1/(A+B+R0-Rl)**2.0          #dtaudR0
        jac[6,6] = 1.0                              #ddtddt
        
        # use the Jacobian to populate the rest of the covariance matrix
        jact = np.transpose(jac)
        pcov_out = np.dot(jac,np.dot(pcov_in,jact))
        
    elif (poles==3):
        ##three poles
        # TODO: convert the 3 pole fit parameters to 3 pole TES parameters
        popt_out = popt
        pcov_out = pcov

    return popt_out,pcov_out

def FitYFreqPriors(freq,dIdV,priors,invpriorsCov,yerr=None,Rl=0.35,R0=0.130,beta=0.5,l=10.0,L=500.0e-9,tau0=500.0e-6,dt=-10.0e-6):
    """Function to directly fit Irwin's TES parameters (Rl, R0, beta, l, L, tau0, dt) with the knowledge of prior known values any number of the parameters. In order for the degeneracy of the parameters to be broken, at least 2 fit parameters should have priors knowledge. This is usually Rl and R0, as these can be known from IV data.
    
    Args:
        freq: Frequencies corresponding to the dIdV
        dIdV: Complex impedance extracted from the trace in frequency space
        priors: Prior known values of Irwin's TES parameters for the trace. Should be in the order of (Rl,R0,beta,l,L,tau0,dt)
        invpriorsCov: Inverse of the covariance matrix of the prior known values of Irwin's TES parameters for the trace (any values that are set to zero mean that we have no knowledge of that 
        yerr: Error at each frequency of the dIdV. Should be a complex number, e.g. yerr = yerr_real + 1.0j * yerr_imag, where yerr_real is the standard deviation of the real part of the dIdV, and yerr_imag is the standard deviation of the imaginary part of the dIdV
        Rl: Guess of the load resistance of the TES circuit (in Ohms)
        R0: Guess of the resistance of the TES (in Ohms)
        beta: Guess of the current sensitivity beta
        l: Guess of Irwin's loop gain
        L: Guess of the inductance (in Henrys)
        tau0: Guess of the thermal time constant (in s)
        dt: Guess of the time shift (in s)
        
    Returns:
        popt: The fitted parameters in the order of (Rl, R0, beta, l, L, tau0, dt)
        pcov: The corresponding covariance matrix for the fitted parameters
        cost: The cost of the the fit
        
    """
    
    p0=(Rl,R0,beta,l,L,tau0,dt)
    bounds=((0.0,0.0,0.0,0.0,0.0,0.0,-1.0e-3),(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1.0e-3))
    
    def residualPriors(params,priors,invpriorsCov):
        # priors = prior known values of Rl, R0, beta, l, L, tau0 (2-pole)
        # invpriorsCov = inverse of the covariance matrix of the priors
        
        z1dpriors = np.sqrt((priors-params).dot(invpriorsCov).dot(priors-params))
        return z1dpriors
        
    def residual(params):
        # define a residual for the nonlinear least squares algorithm 
        fZ=freq
        # different functions for different amounts of poles
        Rl,R0,beta,l,L,tau0,dt=params
        ci=TwoPoleAdmittancePriors(fZ,Rl,R0,beta,l,L,tau0) * np.exp(-2.0j*pi*fZ*dt)
        
        # the difference between the data and the fit
        diff=dIdV-ci
        # get the weights from yerr, these should be 1/(standard deviation) for real and imaginary parts
        if(yerr is None):
            weights=1.0+1.0j
        else:
            weights=1.0/yerr.real+1.0j/yerr.imag
        
        # create the residual vector, splitting up real and imaginary parts of the residual separately
        z1d = np.zeros(fZ.size*2+1, dtype = np.float64)
        z1d[0:z1d.size-1:2] = diff.real*weights.real
        z1d[1:z1d.size-1:2] = diff.imag*weights.imag
        z1d[-1] = residualPriors(params,priors,invpriorsCov)
        return z1d

    def jaca(params):
        # analytically calculate the Jacobian for 2 pole and three pole cases
        popt = params

        # popt = Rl,R0,beta,l,L,tau0,dt
        Rl = popt[0]
        R0 = popt[1]
        beta = popt[2]
        l = popt[3]
        L = popt[4]
        tau0 = popt[5]
        dt = popt[6]
        
        # derivative of 1/x = -1/x**2 (without doing chain rule)
        deriv1 = -1.0/(2.0j*pi*freq*L + Rl + R0*(1.0+beta) + R0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l)))**2
        
        dYdRl = np.zeros(freq.size*2, dtype = np.float64)
        dYdRlcomplex = deriv1 * np.exp(-2.0j*pi*freq*dt)
        dYdRl[0:dYdRl.size:2] = np.real(dYdRlcomplex)
        dYdRl[1:dYdRl.size:2] = np.imag(dYdRlcomplex)

        dYdR0 = np.zeros(freq.size*2, dtype = np.float64)
        dYdR0complex = deriv1 * (1.0+beta + l * (2.0+beta)/(1.0 - l +2.0j*pi*freq*tau0))  * np.exp(-2.0j*pi*freq*dt)
        dYdR0[0:dYdR0.size:2] = np.real(dYdR0complex)
        dYdR0[1:dYdR0.size:2] = np.imag(dYdR0complex)

        dYdbeta = np.zeros(freq.size*2, dtype = np.float64)
        dYdbetacomplex = deriv1 * (R0+2.0j*pi*freq*R0*tau0)/(1.0-l + 2.0j*pi*freq*tau0) * np.exp(-2.0j*pi*freq*dt)
        dYdbeta[0:dYdbeta.size:2] = np.real(dYdbetacomplex)
        dYdbeta[1:dYdbeta.size:2] = np.imag(dYdbetacomplex)

        dYdl = np.zeros(freq.size*2, dtype = np.float64)
        dYdlcomplex = deriv1 * R0*(2.0+beta)*(1.0+2.0j*pi*freq*tau0)/(1.0-l+2.0j*pi*freq*tau0)**2 * np.exp(-2.0j*pi*freq*dt)
        dYdl[0:dYdl.size:2] = np.real(dYdlcomplex)
        dYdl[1:dYdl.size:2] = np.imag(dYdlcomplex)

        dYdL = np.zeros(freq.size*2, dtype = np.float64)
        dYdLcomplex = deriv1 * 2.0j*pi*freq * np.exp(-2.0j*pi*freq*dt)
        dYdL[0:dYdL.size:2] = np.real(dYdLcomplex)
        dYdL[1:dYdL.size:2] = np.imag(dYdLcomplex)

        dYdtau0 = np.zeros(freq.size*2, dtype = np.float64)
        dYdtau0complex = deriv1 * -2.0j*pi*freq*l*R0*(2.0+beta)/(1.0-l+2.0j*pi*freq*tau0)**2 * np.exp(-2.0j*pi*freq*dt)
        dYdtau0[0:dYdtau0.size:2] = np.real(dYdtau0complex)
        dYdtau0[1:dYdtau0.size:2] = np.imag(dYdtau0complex)
        
        dYddt = np.zeros(freq.size*2, dtype = np.float64)
        dYddtcomplex = -2.0j*pi*freq/(2.0j*pi*freq*L + Rl + R0*(1.0+beta) + R0*l*(2.0+beta)/(1.0-l)*1.0/(1.0+2.0j*pi*freq*tau0/(1-l))) * np.exp(-2.0j*pi*freq*dt)
        dYddt[0:dYddt.size:2] = np.real(dYddtcomplex)
        dYddt[1:dYddt.size:2] = np.imag(dYddtcomplex)

        jac = np.column_stack((dYdRl,dYdR0,dYdbeta,dYdl,dYdL,dYdtau0,dYddt))
        return jac

    res=least_squares(residual, p0, bounds=bounds, loss='linear', max_nfev=1000, verbose=0, x_scale=np.abs(p0))
    
    popt=res['x']
    cost=res['cost']
    
    # check if the fit failed (usually only happens when we reach maximum evaluations, likely when fitting assuming the wrong loop gain)
    if(not res['success']):
        print('Fit failed: '+str(res['status']))

    # analytically calculate the covariance matrix
    if(yerr is None):
        weights=1.0+1.0j
    else:
        weights=1.0/yerr.real+1.0j/yerr.imag
    
    #convert weights to variances (want 1/var, as we are creating the inverse of the covariance matrix)
    weightVals = np.zeros(freq.size*2, dtype = np.float64)
    weightVals[0:weightVals.size:2] = weights.real**2
    weightVals[1:weightVals.size:2] = weights.imag**2
    
    jac = jaca(popt)
    jact = np.transpose(jac)
    wjac = np.zeros_like(jac)
    
    # right multiply inverse of covariance matrix by the jacobian (we do this element by element, to avoid creating a huge covariance matrix)
    for ii in range(0,len(popt)):
        wjac[:,ii] = np.multiply(weightVals,jac[:,ii])
        
    # left multiply by the jacobian and take the inverse to get the analytic covariance matrix
    pcovinv = np.dot(jact,wjac) + invpriorsCov
    pcov = np.linalg.inv(pcovinv)
    
    return popt,pcov,cost

def ConvertFromTESValues(popt,pcov):
    """Function to convert from Irwin's TES parameters (Rl, R0, beta, l, L, tau0, dt) to the fit parameters (A, B, tau1, tau2, dt)
    
    Args:
        popt: Irwin's TES parameters in the order of (Rl, R0, beta, l, L, tau0, dt), should be a 1-dimensional np.array of length 7
        pcov: The corresponding covariance matrix for Irwin's TES parameters. Should be a 2-dimensional, 7-by-7 np.array
        
    Returns:
        popt_out: The fit parameters in the order of (A, B, tau1, tau2, dt)
        pcov_out: The corresponding covariance matrix for the fit parameters
        
    """
   
    ## two poles
    # extract fit parameters
    Rl = popt[0]
    R0 = popt[1]
    beta = popt[2]
    l = popt[3]
    L = popt[4]
    tau0 = popt[5]
    dt = popt[6]
    
    # convert A, B tau1, tau2 to beta, l, L, tau
    A = Rl + R0 * (1.0+beta)
    B = R0 * l/(1.0-l) * (2.0+beta)
    tau1 = tau0/(1.0-l)
    tau2 = L/(Rl+R0*(1.0+beta))
    
    popt_out = [A,B,tau1,tau2,dt]

    # calculate the Jacobian
    jac = np.zeros((5,7))
    jac[0,0] = 1.0        #dAdRl
    jac[0,1] = 1.0 + beta #dAdR0
    jac[0,2] = R0         #dAdbeta
    jac[1,1] = l/(1.0-l) * (2.0+beta) #dBdR0
    jac[1,2] = l/(1.0-l) * R0 #dBdbeta
    jac[1,3] = R0 * (2.0+beta)/(1.0-l)  + l/(1.0-l)**2.0 * R0 * (2.0+beta) #dBdl
    jac[2,3] = tau0/(1.0-l)**2.0  #dtau1dl
    jac[2,5] = 1.0/(1.0-l) #dtau1dtau0
    jac[3,0] = -L/(Rl+R0*(1.0+beta))**2.0 #dtau2dRl
    jac[3,1] = -L * (1.0+beta)/(Rl+R0*(1.0+beta))**2 #dtau2dR0
    jac[3,2] = -L*R0/(Rl+R0*(1.0+beta))**2.0 #dtau2dbeta
    jac[3,4] = 1.0/(Rl+R0*(1.0+beta))#dtau2dL
    jac[4,6] = 1.0 #ddtddt
    

    # use the Jacobian to populate the rest of the covariance matrix
    jact = np.transpose(jac)
    pcov_out = np.dot(jac,np.dot(pcov,jact))
        

    return popt_out,pcov_out

def FindPoleFallTimes(params):
    """Function for taking TES params from a 1-pole, 2-pole, or 3-pole dIdV and calculating the falltimes (i.e. the values of the poles in the complex plane)
    
    Args:
        params: TES parameters for either 1-pole, 2-pole, or 3-pole dIdV. This will be a 1-dimensional np.array of varying length, depending on the fit. 1-pole fit has 3 parameters (A,tau2,dt), 2-pole fit has 5 parameters (A,B,tau1,tau2,dt), and 3-pole fit has 7 parameters (A,B,C,tau1,tau2,tau3,dt). The parameters should be in that order, and any other number of parameters will print a warning and return zero.
        
    Returns:
        np.sort(fallTimes): The falltimes for the dIdV fit, sorted from fastest to slowest.
        
    """
    
    # convert dVdI time constants to fall times of dIdV
    if len(params)==3:
        # one pole fall time for dIdV is same as tau2=L/R
        A,tau2,dt = params
        fallTimes = np.array([tau2])
        
    elif len(params)==5:
        # two pole fall times for dIdV is different than tau1, tau2
        A,B,tau1,tau2,dt = params
        
        def TwoPoleEquations(p):
            taup,taum = p
            eq1 = taup+taum - A/(A+B)*(tau1+tau2)
            eq2 = taup*taum-A/(A+B)*tau1*tau2
            return (eq1,eq2)
        
        taup,taum = fsolve(TwoPoleEquations,(tau1,tau2))
        fallTimes = np.array([taup,taum])
        
    elif len(params)==7:
        # three pole fall times for dIdV is different than tau1, tau2, tau3
        A,B,C,tau1,tau2,tau3,dt = params
        
        def ThreePoleEquations(p):
            taup,taum,taun = p
            eq1 = taup+taum+taun-(A*tau1+A*(1.0-C)*tau2+(A+B)*tau3)/(A*(1.0-C)+B)
            eq2 = taup*taum+taup*taun+taum*taun - (tau1*tau2+tau1*tau3+tau2*tau3)*A/(A*(1.0-C)+B)
            eq3 = taup*taum*taun - tau1*tau2*tau3*A/(A*(1.0-C)+B)
            return (eq1,eq2,eq3)
        
        taup,taum,taun = fsolve(ThreePoleEquations,(tau1,tau2,tau3))
        fallTimes = np.array([taup,taum,taun])
        
    else:
        print "Wrong number of input parameters, returning zero..."
        fallTimes = np.zeros(1)
    
    # return fall times sorted from shortest to longest
    return np.sort(fallTimes)

def DeconvolveDIDV(x,trace,Rsh,sgAmp,sgFreq,dutycycle):
    """Function for taking a trace with a known square wave jitter and extracting the complex impedance via deconvolution of the square wave and the TES response in frequency space.
    
    Args:
        x: Time values for the trace
        trace: The trace in time domain (in Amps)
        Rsh: Shunt resistance for electronics (in Ohms)
        sgAmp: Peak to peak value of square wave jitter (in Amps)
        sgFreq: Frequency of square wave jitter
        dutycycle: duty cycle of square wave jitter
        
    Returns:
        freq: The frequencies that each point of the trace corresponds to
        dIdV: Complex impedance of the trace in frequency space
        zeroInds: Indices of the frequencies where the trace's Fourier Transform is zero. Since we divide by the FT of the trace, we need to know which values should be zero, so that we can ignore these points in the complex impedance.
        
    """
    
    tracelength=len(x)
    
    # get the frequencies for a DFT, based on the sample rate of the data
    dx=x[1]-x[0]
    freq=fftfreq(len(x),d=dx)
    
    # FFT of the trace
    St = fft(trace)
    
    # analytic DFT of a duty cycled square wave
    Sf = np.zeros_like(freq)*0.0j
    
    # even frequencies are zero unless the duty cycle is not 0.5
    if (dutycycle==0.5):
        oddInds = ((np.abs(np.mod(np.absolute(freq/sgFreq),2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = 1.0j/(pi*freq[oddInds]/sgFreq)*sgAmp*Rsh*tracelength
    else:
        oddInds = ((np.abs(np.mod(np.abs(freq/sgFreq),2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        Sf[oddInds] = -1.0j/(2.0*pi*freq[oddInds]/sgFreq)*sgAmp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[oddInds]/sgFreq*dutycycle)-1)
        
        evenInds = ((np.abs(np.mod(np.abs(freq/sgFreq)+1,2)-1))<1e-8) #due to float precision, np.mod will have errors on the order of 1e-10 for large numbers, thus we set a bound on the error (1e-8)
        evenInds[0] = False
        Sf[evenInds] = -1.0j/(2.0*pi*freq[evenInds]/sgFreq)*sgAmp*Rsh*tracelength*(np.exp(-2.0j*pi*freq[evenInds]/sgFreq*dutycycle)-1)
    
    # the tracelength/2 value from the FFT is purely real, which can cause errors when taking the standard deviation (get stddev = 0 for real part of dIdV at this frequency, leading to a divide by zero when calculating the residual when fitting)
    Sf[tracelength/2]=0.0j
    
    # deconvolve the trace from the square wave to get the dIdV in frequency space
    dVdI=(Sf/St)
    
    # set values that are within floating point error of zero to 1.0 + 1.0j (we will give these values virtually infinite error, so the value doesn't matter. Setting to 1.0+1.0j avoids divide by zero if we invert)
    zeroInds=np.abs(dVdI) < 1e-16
    dVdI[zeroInds]=(1.0+1.0j)
    
    # convert to complex admittance
    dIdV=1.0/dVdI

    return freq,dIdV,zeroInds

def get_values(trace, flatTimes, flatInds, timeArray, Rsh, sgAmp, sgFreq, dutycycle):
    """Function for getting all relevant parameters of a single trace - range, slope, mean, skewness. Also calculates the complex impedance of the trace.
    
    Args:
        trace: A single trace to be analyzed
        flatTimes: np.ndarray of times at which the dIdV curve is flat
        flatInds: Indices of the trace where the dIdV curve is flat
        timeArray: The corresponding time for each data point in trace (in s)
        Rsh: Shunt resistance for electronics (in Ohms)
        sgAmp: Peak to peak value of square wave jitter
        sgFreq: Frequency of square wave jitter
        dutycycle: duty cycle of square wave jitter
        
    Returns:
        traceRange: The range of the trace (max - min)
        traceSlope: The slope of the trace
        traceMean: The mean value of the trace
        traceSkewness: The skewness of the trace
        dIdVi: The complex impedance of the TES in frequency space, extracted from the trace
        
    """
    traceRange = max(trace)-min(trace)
    traceSlope = slope(flatTimes,trace[flatInds])
    traceMean = np.mean(trace)
    traceSkewness = skew(trace)
    dIdVi = DeconvolveDIDV(timeArray,trace,Rsh,sgAmp,sgFreq,dutycycle)[1]
    return [traceRange, traceSlope, traceMean, traceSkewness, dIdVi]

def get_values_star(args):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return get_values(*args)

def processDIDV(rawTraces, timeOffset=0, traceGain=1.25e5, sgFreq=200.0, sgAmp=0.25, fs=2e5, dutycycle=0.5, add180Phase=False, fit=False, autoCut=False, pathSave='', fileStr='', makePlots=False, saveResults=False, priors=None, invpriorsCov=None, R0=85e-3, dR0=10e-3, Rp=10e-3, dRp=3e-3, Rsh=24e-3, dt0=10.0e-6):
    """Function for processing a DIDV curve, assuming a square wave jitter. This function takes an array of traces, takes the mean, switches to frequency space, finds the complex impedance, and fits it to various models. The function supports 1, 2, and 3 pole fits. A 2 pole fit with known priors is also supported, as long as the priors and the inverse of the priors covariance are provided.
    
    Args:
        rawTraces: Array of all of the traces to be analyzed, expected to be stored in a (number of traces)x(length of trace) sized array
        timeOffset: Adds a time offset (in seconds) to match the FFT to a the start of a period
        traceGain: Value to divide the units of rawTraces by in order to convert to Amps
        sgFreq: Frequency of square wave jitter
        sgAmp: Peak to peak value of square wave (in Amps)
        fs: Sample frequency of data
        dutycycle: Duty cycle of square wave jitter
        add180Phase: Algorithm expects a period of square wave to look like: __-- (i.e. lower part first, then higer end), if this is not the case, use this flag to add a half-period time shift to match this expectation
        fit: Flag to do the fit or not
        autoCut: Flag to apply auto cuts to the data or not
        pathSave: Path to save the outputted data/plots
        fileStr: Filename to be used when saving outputted data/plots
        makePlots: Flag to make and save the plots
        saveResults: Flag to save the results (e.g. the fit parameters)
        R0: Resistance of TES from other data (e.g. IV plots) (in Ohms)
        dR0: Standard deviation associated with R0 (in Ohms)
        Rp: Parasitic resistance of TES (in Ohms)
        dRp: Standard deviation associated with Rp (in Ohms)
        Rsh: Shunt resistance (in Ohms)
        priors: Prior known values of TES params in 2 pole fit, should be in order of (Rl,R0,beta,l,L,tau0,dt)
        invpriorsCov: Inverse of the covariance matrix of the prior values (only the diagonal is read at this time)
        dt0: Initial guess for the time offset correction (in s). If the initial guess is far from the true offset, then the fit can converge to an inaccurate value. It is recommended that the user run the fit a few times, iteratively changing the time offset until it converges to the correct value, then using that value for all traces.
        
    Returns:
        savedData: Dictionary that stores the result of the algorithm.
        
    """
    if rawTraces.dtype == "object":
        rawTraces = np.vstack(rawTraces) # in case input data is an array of numpy arrays (this can happen if using the getRawEvents function for IO)
    
    #get number of traces 
    nTraces = len(rawTraces)
    #converting sampling rate to time step
    dt=(1.0/fs) 
    
    #get trace x values (i.e. time) in seconds
    bins=np.arange(0,len(rawTraces[0]))
    
    # add half a period of the square wave frequency if add180Phase is True
    if (add180Phase):
        timeOffset = timeOffset + 1/(2*sgFreq)
    
    # apply time offset
    timeArray=bins*dt-timeOffset
    indOffset=int(timeOffset*fs)
    
    #figure out how many dIdV periods are in the trace, including the time offset
    period=1.0/sgFreq
    nPeriods = np.floor((max(timeArray)-timeArray[indOffset])/period)
    
    # find which indices to keep in order to have an integer number of periods, as well as the inputted timeoffset
    indMax = int(nPeriods*fs/sgFreq)
    good_inds = range(indOffset,indMax+indOffset)
    
    # ignore the tail of the trace after the last period, as this tail just adds artifacts to the FFTs
    timeArray = timeArray[good_inds]
    traces=rawTraces[:,good_inds]/(traceGain) # convert to Amps

    #need these x-values to be properly scaled for maximum likelihood slope fitting
    period_unscaled=fs/sgFreq
    
    #save the  "top slope" points in the trace, which are the points just before the overshoot in the dI/dV
    flatIndsTemp=list()
    for i in range(0,int(nPeriods)):
        # get index ranges for flat parts of trace
        flatIndLow=int((float(i)+0.25)*period_unscaled)
        flatIndHigh=int((float(i)+0.48)*period_unscaled)
        flatIndsTemp.append(range(flatIndLow,flatIndHigh))
    flatInds=np.array(flatIndsTemp).flatten()
    flatTimes=timeArray[flatInds]
    
#     n_processes = mp.cpu_count()
#     pool = mp.Pool(processes=n_processes)
    
#     itervalues = izip(traces, repeat(flatTimes), repeat(flatInds), repeat(timeArray), repeat(Rsh), repeat(sgAmp), repeat(sgFreq), repeat(dutycycle))
    
#     values = pool.map(get_values_star,itervalues)
#     pool.terminate()
#     pool.close()
#     pool.join()
    
#     ranges = np.array(values,dtype=object)[:,0].astype(float)
#     slopes = np.array(values,dtype=object)[:,1].astype(float)
#     means = np.array(values,dtype=object)[:,2].astype(float)
#     skewnesses = np.array(values,dtype=object)[:,3].astype(float)
#     dIdVs = np.vstack(np.array(values,dtype=object)[:,4])
    
#     if(autoCut):
#         slopes_all=slopes
#         means_all=means
#         skews_all=skewnesses
#         ranges_all=ranges
        
#         cut = autoCuts(traces,traceGain=1.0,fs=fs,isDIDV=True,sgFreq=sgFreq)
        
#         means=means[cut]
#         ranges=ranges[cut]
#         slopes=slopes[cut]
#         skewnesses=skewnesses[cut]
#         traces=traces[cut]
#         dIdVs=dIdVs[cut]


    #for storing results
    skewnesses=list()
    means=list()
    ranges=list()
    slopes=list()
    traces=list()
    dIdVs=list()

    for rawTrace in rawTraces:
        # store all traces, converted to amps, as tt
        trace=rawTrace[good_inds]/(traceGain) #ignore the tail of the trace after the last period, as this tail just adds artifacts to the FFTs
        
        # store the ranges
        ranges.append(max(trace)-min(trace))
        
        # store the slopes (total slope of the trace)
        topSlope = slope(flatTimes,trace[flatInds])
        slopes.append(topSlope)
        
        # store the means of the traces, to use for finding the offset
        means.append(np.mean(trace))
        
        # store the skewnesses
        skewness=skew(trace)
        skewnesses.append(abs(skewness))
        
        # append the trace to the list of all traces
        traces.append(trace)
        
        # deconvolve the trace from the square wave to get the dI/dV in frequency domain
        fdIdVi,dIdVi,zeroInds = DeconvolveDIDV(timeArray,trace,Rsh,sgAmp,sgFreq,dutycycle)
        dIdVs.append(dIdVi)
    
    #convert to numpy structures
    traces=np.array(traces)
    dIdVs=np.array(dIdVs)
    means=np.array(means)
    skewnesses=np.array(skewnesses)
    slopes=np.array(slopes)
    ranges=np.array(ranges)
    
    #store results
    tmean=np.mean(traces,axis=0)
    fdIdV,dIdV_mean,zeroInds = DeconvolveDIDV(timeArray,tmean,Rsh,sgAmp,sgFreq,dutycycle)
    
    # divide by sqrt(N) for standard deviation of mean
    sdIdV=stdComplex(dIdVs)/np.sqrt(nTraces)
    sdIdV[zeroInds] = (1.0+1.0j)*1.0e20
    dIdV=np.mean(dIdVs,axis=0)
    
    offset=np.mean(means)
    doffset=np.std(means)/np.sqrt(nTraces)
    
    Rl=Rp+Rsh # Rload is defined as shunt resistance (Rsh) plus parasitic resistance (Rp)
    
    if(fit):
        # guess the 1 pole square wave parameters
        A0_1pole,tau20_1pole = SquareWaveGuessParams(tmean,sgAmp,Rsh)
        
        # Guess the starting parameters for 2 pole fitting
        A0,B0,tau10,tau20,isLoopGainSub1 = GuessDIDVParams(tmean,tmean[flatInds],sgAmp,Rsh,L0=1.0e-7)
        
        # 1 pole fitting
        v1,s1,cost1 = FitYFreq(fdIdV,dIdV,yerr=sdIdV,A0=A0_1pole,tau20=tau20_1pole,dt=dt0,poles=1)
        yFit1 = YDI(timeArray,v1[0],0.0,0.0,0.0,v1[1],0.0,sgAmp,Rsh,sgFreq,dutycycle)+offset
        
        # 2 pole fitting
        v2,s2,cost2 = FitYFreq(fdIdV,dIdV,yerr=sdIdV,A0=A0,B0=B0,tau10=tau10,tau20=tau20,dt=dt0,poles=2)
        yFit2 = YDI(timeArray,v2[0],v2[1],0.0,v2[2],v2[3],0.0,sgAmp,Rsh,sgFreq,dutycycle)+offset
        
        # 3 pole fitting
        v3,s3,cost3 = FitYFreq(fdIdV,dIdV,yerr=sdIdV,A0=v2[0],B0=-abs(v2[1]),C0=-0.01,tau10=-abs(v2[2]),tau20=v2[3],tau30=1.0e-4,dt=v2[4],poles=3)
        yFit3 = YDI(timeArray,v3[0],v3[1],v3[2],v3[3],v3[4],v3[5],sgAmp,Rsh,sgFreq,dutycycle)+offset
        
        # Convert parameters from 1 and 2 pole fits to the Irwin parameters
        popt_out1,pcov_out1 = ConvertToTESValues(v1,s1,R0,Rl,dR0=dR0,dRl=dRp,poles=1) # 1 pole params (Rtot,L,R0,Rl,dt)
        popt_out2,pcov_out2 = ConvertToTESValues(v2,s2,R0,Rl,dR0=dR0,dRl=dRp,poles=2) # 2 pole params (beta, l, L, tau0, R0, Rl, dt)
        
        # Convert to dIdV falltimes
        OnePoleFallTimes = FindPoleFallTimes(v1)
        TwoPoleFallTimes = FindPoleFallTimes(v2)
        ThreePoleFallTimes = FindPoleFallTimes(v3)
        
        fFit = fdIdV
        
        if priors is not None:
            # convert guesses to Rl, R0, beta, l, L, tau0 basis
#            guesspriors = ConvertToTESValues([A0,B0,tau10,tau20,1.0e-6],np.diag(np.ones(5)),R0,Rl,dR0=dR0,dRl=dRp,poles=2)[0] # 2 pole params (beta, l, L, tau0, R0, Rl, dt)
            
            # each guess should be positive
            beta0 = abs(popt_out2[2])
            l0 = abs(popt_out2[3])
            L0 = abs(popt_out2[4])
            tau0 = abs(popt_out2[5])
            
            # 2 pole fitting
            v2priors,s2priors,costPriors = FitYFreqPriors(fdIdV,dIdV,priors,invpriorsCov,yerr=sdIdV,R0=abs(R0),Rl=abs(Rl),beta=beta0,l=l0,L=L0,tau0=tau0,dt=popt_out2[6])
            
            # convert answer back to A, B, tauI, tauEL basis for plotting
            v2priorsConv = ConvertFromTESValues(v2priors,s2priors)[0]
            
            # Find the dIdV falltimes
            TwoPoleFallTimesPriors = FindPoleFallTimes(v2priorsConv)
            
            # save the fits with priors in time and frequency domain
            yFit2priors = YDI(timeArray,v2priorsConv[0],v2priorsConv[1],0.0,v2priorsConv[2],v2priorsConv[3],0.0,sgAmp,Rsh,sgFreq,dutycycle)+offset
            dIdVFit2priors = TwoPoleAdmittancePriors(fFit,v2priors[0],v2priors[1],v2priors[2],v2priors[3],v2priors[4],v2priors[5]) * np.exp(-2.0j*pi*fFit*v2priors[6])
        else:
            # set the priors variables to None
            v2priors = None
            s2priors = None
            yFit2priors = None
            dIdVFit2priors = None
            TwoPoleFallTimesPriors=None
            
        ## save the fits in frequency domain as variables for saving/plotting
        dIdVFit1 = OnePoleAdmittance(fFit,v1[0],v1[1]) * np.exp(-2.0j*pi*fFit*v1[2])
        dIdVFit2 = TwoPoleAdmittance(fFit,v2[0],v2[1],v2[2],v2[3]) * np.exp(-2.0j*pi*fFit*v2[4])
        dIdVFit3 = ThreePoleAdmittance(fFit,v3[0],v3[1],v3[2],v3[3],v3[4],v3[5]) * np.exp(-2.0j*pi*fFit*v3[6])

    if(saveResults):
        with h5py.File(pathSave+'dIdV'+fileStr+'.h5',"w") as f:
            f.create_dataset("t",data=timeArray)
            f.create_dataset("trace_mean", data=tmean)
            f.create_dataset("freq", data=fdIdV)
            f.create_dataset("dIdV_mean", data=dIdV)
            f.create_dataset("sdIdV", data=sdIdV)
            f.attrs["Is0"] = offset
            f.attrs["dIs0"] = doffset
            f.attrs["nTraces"] = nTraces
            f.attrs["fs"] = fs
            f.attrs["sgAmp"] = sgAmp
            f.attrs["sgFreq"] = sgFreq
            f.attrs["dutycycle"] = dutycycle
            
            if fit:
                f.create_dataset("yfit_freq_p1", data=yFit1)
                f.create_dataset("yfit_freq_p2", data=yFit2)
                f.create_dataset("yfit_freq_p3", data=yFit3)
                f.create_dataset("dIdVfit_freq_p1", data=dIdVFit1)
                f.create_dataset("dIdVfit_freq_p2", data=dIdVFit2)
                f.create_dataset("dIdVfit_freq_p3", data=dIdVFit3)
                f.create_dataset("yfit_freq_p2Priors", data=yFit2priors)
                f.create_dataset("dIdVfit_freq_p2Priors", data=dIdVFit2priors)
                f.attrs["CIparams_freq_p1"] = v1
                f.attrs["CIcov_freq_p1"] = s1
                f.attrs["CIcost_freq_p1"] = cost1
                f.attrs["CIparams_freq_p2"] = v2
                f.attrs["CIcov_freq_p2"] = s2
                f.attrs["CIcost_freq_p2"] = cost2
                f.attrs["CIparams_freq_p3"] = v3
                f.attrs["CIcov_freq_p3"] = s3
                f.attrs["CIcost_freq_p3"] = cost3
                f.attrs["OnePoleTESparams"] = popt_out1
                f.attrs["OnePoleTEScov"] = pcov_out1
                f.attrs["TwoPoleTESparams"] = popt_out2
                f.attrs["TwoPoleTEScov"] = pcov_out2
                f.attrs["TwoPoleTESparamsPriors"] = v2priors
                f.attrs["TwoPoleTEScovPriors"] = s2priors
                f.attrs["TwoPoleTEScostPriors"] = costPriors
                f.attrs["OnePoleFallTimes"] = OnePoleFallTimes
                f.attrs["TwoPoleFallTimes"] = TwoPoleFallTimes
                f.attrs["ThreePoleFallTimes"] = ThreePoleFallTimes
                f.attrs["TwoPoleFallTimesPriors"] = TwoPoleFallTimesPriors
    if(fit):
        savedData = {'t':timeArray,
                    'y':tmean,
                    'yfit_freq_p1':yFit1,
                    'yfit_freq_p2':yFit2,
                    'yfit_freq_p3':yFit3,
                    'fFit':fFit,
                    'dIdVfit_freq_p1':dIdVFit1,
                    'dIdVfit_freq_p2':dIdVFit2,
                    'dIdVfit_freq_p3':dIdVFit3,
                    'CIparams_freq_p1':v1,
                    'CIcov_freq_p1':s1,
                    'CIcost_freq_p1':cost1,
                    'CIparams_freq_p2':v2,
                    'CIcov_freq_p2':s2,
                    'CIcost_freq_p2':cost2,
                    'CIparams_freq_p3':v3,
                    'CIcov_freq_p3':s3,
                    'CIcost_freq_p3':cost3,
                    'dIdV':dIdV,
                    'dIdVmean':dIdV_mean,
                    'sdIdV':sdIdV,
                    'fdIdV':fdIdV,
                    'Is0':offset,
                    'dIs0':doffset,
                    'OnePoleTESparams':popt_out1,
                    'OnePoleTEScov':pcov_out1,
                    'TwoPoleTESparams':popt_out2,
                    'TwoPoleTEScov':pcov_out2,
                    'TwoPoleTESparamsPriors':v2priors,
                    'TwoPoleTEScovPriors':s2priors,
                    "TwoPoleTEScostPriors":costPriors,
                    'yfit_freq_p2Priors':yFit2priors,
                    'dIdVfit_freq_p2Priors':dIdVFit2priors,
                    'OnePoleFallTimes':OnePoleFallTimes,
                    'TwoPoleFallTimes':TwoPoleFallTimes,
                    'ThreePoleFallTimes':ThreePoleFallTimes,
                    'TwoPoleFallTimesPriors':TwoPoleFallTimesPriors,
                    'nTraces': nTraces}
    else:
        savedData = {'t':timeArray,
                    'y':tmean,
                    'dIdV':dIdV,
                    'dIdVmean':dIdV_mean,
                    'sdIdV':sdIdV,
                    'fdIdV':fdIdV,
                    'Is0':offset,
                    'dIs0':doffset,
                    'nTraces': nTraces}
    if(makePlots):
        
        ## plot the entire trace with fits
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(timeArray*1e6,tmean*1e6,color='black',label='mean')
        ax.scatter(timeArray[flatInds]*1e6,tmean[flatInds]*1e6,color='blue',marker='.',label='Slope Points',zorder=5)
        if(fit):
            ax.plot(timeArray*1e6+v1[2]*1e6,yFit1*1e6,'-',color='magenta',alpha=0.9,label='x(f) 1-pole fit')
            ax.plot(timeArray*1e6+v2[4]*1e6,yFit2*1e6,'-',color='green',alpha=0.9,label='x(f) 2-pole fit')
            ax.plot(timeArray*1e6+v3[6]*1e6,yFit3*1e6,'-',color='orange',alpha=0.9,label='x(f) 3-pole fit')
            if priors is not None:
                ax.plot(timeArray*1e6+v2priors[6]*1e6,yFit2priors*1e6,'-',color='cyan',alpha=0.9,label='x(f) 2-pole fit with priors')
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')
        ax.set_xlim(0,max(timeArray)*1e6)
        
        ymax=max(tmean)-offset
        ymin=offset-min(tmean)
        urange=max([ymax,ymin])
        ax.set_ylim((offset-urange*1.5)*1e6,(offset+urange*1.5)*1e6)
        ax.legend(loc='upper left')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIsTraces'+fileStr+'.png')
        plt.close(fig)
        
        ## plot a single period of the trace
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(timeArray*1e6,tmean*1e6,color='black',label='data')
        if(fit):
            ax.plot(timeArray*1e6+v1[2]*1e6,yFit1*1e6,color='magenta',label='x(f) 1-pole fit')
            ax.plot(timeArray*1e6+v2[4]*1e6,yFit2*1e6,color='green',label='x(f) 2-pole fit')
            ax.plot(timeArray*1e6+v3[6]*1e6,yFit3*1e6,color='orange',label='x(f) 3-pole fit')
            if priors is not None:
                ax.plot(timeArray*1e6+v2priors[6]*1e6,yFit2priors*1e6,'-',color='cyan',alpha=0.9,label='x(f) 2-pole fit with priors')
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')

        halfRange=0.6*period
        ax.set_xlim((period-halfRange)*1e6,(period+halfRange)*1e6)

        ax.legend(loc='upper left')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIsTracesFit'+fileStr+'.png')
        plt.close(fig)

        ## plot zoomed in on the trace
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(timeArray*1e6,tmean*1e6,color='black',label='data')
        if(fit):
            ax.plot(timeArray*1e6+v1[2]*1e6,yFit1*1e6,color='magenta',label='x(f) 1-pole fit')
            ax.plot(timeArray*1e6+v2[4]*1e6,yFit2*1e6,color='green',label='x(f) 2-pole fit')
            ax.plot(timeArray*1e6+v3[6]*1e6,yFit3*1e6,color='orange',label='x(f) 3-pole fit')
            if priors is not None:
                ax.plot(timeArray*1e6+v2priors[6]*1e6,yFit2priors*1e6,'-',color='cyan',alpha=0.9,label='x(f) 2-pole fit with priors')
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')

        halfRange=0.1*period
        ax.set_xlim((period-halfRange)*1e6,(period+halfRange)*1e6)

        ax.legend(loc='upper left')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIsTracesZoomFit'+fileStr+'.png')
        plt.close(fig)
        
        ## plot the traces as well as the traces flipped in order to check asymmetry
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(timeArray*1e6,(tmean-offset)*1e6,color='black',label='data')
        timeArray_flipped=timeArray-period/2.0
        tmean_flipped=-(tmean-offset)
        ax.plot(timeArray_flipped*1e6,tmean_flipped*1e6,color='blue',label='flipped data')
        if(fit):
            ax.plot(timeArray*1e6+v1[2]*1e6,(yFit1-offset)*1e6,color='magenta',label='x(f) 1-pole fit')
            ax.plot(timeArray*1e6+v2[4]*1e6,(yFit2-offset)*1e6,color='green',label='x(f) 2-pole fit')
            ax.plot(timeArray*1e6+v3[6]*1e6,(yFit3-offset)*1e6,color='orange',label='x(f) 3-pole fit')
            if priors is not None:
                ax.plot(timeArray*1e6+v2priors[6]*1e6,(yFit2priors-offset)*1e6,'-',color='cyan',alpha=0.9,label='x(f) 2-pole fit with priors')
        ax.set_xlabel('Time ($\mu$s)')
        ax.set_ylabel('Amplitude ($\mu$A)')
        ax.legend(loc='upper left')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIsTracesFlipped'+fileStr+'.png')
        plt.close(fig)
        
        ## histogram of the skewnesses
        fig,ax=plt.subplots(figsize=(8,6))
        ax.hist(skewnesses,bins=200,range=(0,1),log=True,histtype='step',color='green')
        if(autoCut):
            ax.hist(skews_all,bins=200,range=(0,1),log=True,histtype='step',color='black')
        ax.set_xlabel('Skewness')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIdV_Skewness'+fileStr+'.png')
        plt.close(fig)
        
        ## histogram of the means
        fig,ax=plt.subplots(figsize=(8,6))
        topMean=np.max(means*1e6)
        botMean=np.min(means*1e6)
        ax.hist(means*1e6,bins=200,range=(botMean,topMean),log=True,histtype='step',color='green')
        if(autoCut):
            ax.hist(means_all*1e6,bins=200,range=(botMean,topMean),log=True,histtype='step',color='black')
        ax.set_xlabel('Mean ($\mu$A)')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIdV_Mean'+fileStr+'.png')
        plt.close(fig)
        
        ## histogram of the ranges
        fig,ax=plt.subplots(figsize=(8,6))
        if(autoCut):
            rmax=max(ranges_all*1e6)
        else:
            rmax=max(ranges*1e6)
        if(rmax < 10.0):
            rmax=10.0
        ax.hist(ranges*1e6,bins=200,range=(0.0,rmax),log=True,histtype='step',color='green')
        if(autoCut):
            ax.hist(ranges_all*1e6,bins=200,range=(0.0,rmax),log=True,histtype='step',color='black')
        ax.set_xlabel('Range ($\mu$A)')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIdV_Range'+fileStr+'.png')
        plt.close(fig)
        
        ## histogram of the slopes
        fig,ax=plt.subplots(figsize=(8,6))
        ts=np.array(slopes)*1e6
        ax.hist(slopes*1e6,bins=200,range=(-10,10),log=True,histtype='step',label='Top',color='green')
        if(autoCut):
            ax.hist(slopes_all*1e6,bins=200,range=(-10,10),log=True,histtype='step',label='Top',color='black')
        ax.set_xlabel('Slope ($\mu$A/s)')
        ax.set_title(fileStr)
        ax.grid(linestyle='dotted')
        ax.tick_params(which='both',direction='in',right='on',top='on')
        fig.savefig(pathSave+'dIdV_Slope'+fileStr+'.png')
        plt.close(fig)
        
        ## don't plot points with huge errors
        frac_err=sdIdV/dIdV
        plotInds=np.abs(1/frac_err) > 2.0 
        
        ## plot the real part of the dI/dV in frequency domain
        fig,ax=plt.subplots(figsize=(10,6))
        
        ax.plot(fFit[fFit>0],np.real(dIdVFit2)[fFit>0],color='green',label='x(f) 2-pole fit',zorder=7)
        ax.plot(fFit[fFit>0],np.real(dIdVFit3)[fFit>0],color='orange',label='x(f) 3-pole fit',zorder=8)
        if priors is not None:
            ax.plot(fFit[fFit>0],np.real(dIdVFit2priors)[fFit>0],color='cyan',label='x(f) 2-pole fit with priors',zorder=9)
        ax.scatter(fdIdV[plotInds][fdIdV[plotInds]>0],np.real(dIdV[plotInds])[fdIdV[plotInds]>0],color='blue',label='x(f) mean',s=5,zorder=2)
        
        ## plot error in real part of dIdV
        ax.plot(fdIdV[plotInds][fdIdV[plotInds]>0],np.real(dIdV[plotInds]+sdIdV[plotInds])[fdIdV[plotInds]>0],color='black',label='x(f) 1-$\sigma$ bounds',alpha=0.1,zorder=1)
        ax.plot(fdIdV[plotInds][fdIdV[plotInds]>0],np.real(dIdV[plotInds]-sdIdV[plotInds])[fdIdV[plotInds]>0],color='black',alpha=0.1,zorder=1)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Re($dI/dV$) ($\Omega^{-1}$)')
        ax.set_xscale('log')
        ax.set_xlim(min(fdIdV),max(fdIdV))
        ax.set_ylim(-100,100)
        ax.legend(loc='upper left')
        ax.set_title(fileStr)
        ax.tick_params(right='on',top='on')
        ax.grid(which='major')
        ax.grid(which='minor',linestyle='dotted',alpha=0.3)
        
        fig.savefig(pathSave+'dIdV_Real'+fileStr+'.png')
        plt.close(fig)
            
            
        ## plot the imaginary part of the dI/dV in frequency domain
        fig,ax=plt.subplots(figsize=(10,6))
        ax.plot(fFit[fFit>0],np.imag(dIdVFit2)[fFit>0],color='green',label='x(f) 2-pole fit',zorder=7)
        ax.plot(fFit[fFit>0],np.imag(dIdVFit3)[fFit>0],color='orange',label='x(f) 3-pole fit',zorder=8)
        if priors is not None:
            ax.plot(fFit[fFit>0],np.imag(dIdVFit2priors)[fFit>0],color='cyan',label='x(f) 2-pole fit with priors',zorder=9)
        ax.scatter(fdIdV[plotInds][fdIdV[plotInds]>0],np.imag(dIdV[plotInds])[fdIdV[plotInds]>0],color='blue',label='x(f) mean',s=5,zorder=2)
        
        ## plot error in imaginary part of dIdV
        ax.plot(fdIdV[plotInds][fdIdV[plotInds]>0],np.imag(dIdV[plotInds]+sdIdV[plotInds])[fdIdV[plotInds]>0],color='black',label='x(f) 1-$\sigma$ bounds',alpha=0.1,zorder=1)
        ax.plot(fdIdV[plotInds][fdIdV[plotInds]>0],np.imag(dIdV[plotInds]-sdIdV[plotInds])[fdIdV[plotInds]>0],color='black',alpha=0.1,zorder=1)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Im($dI/dV$) ($\Omega^{-1}$)')
        ax.set_xscale('log')
        ax.set_xlim(min(fdIdV),max(fdIdV))
        ax.set_ylim(-100,100)
        ax.legend(loc='upper left')
        ax.set_title(fileStr)
        ax.tick_params(which='both',direction='in',right='on',top='on')
        ax.grid(which='major')
        ax.grid(which='minor',linestyle='dotted',alpha=0.3)
        
        fig.savefig(pathSave+'dIdV_Imag'+fileStr+'.png')
        plt.close(fig)

    return savedData
