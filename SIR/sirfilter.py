import numpy as np

# Simple implementation of sequential importance filter for 
# observed drifter positions.
#
# kaihc@met.no - 20170329

def getWeights(xpos, ypos, xobs, yobs, R):

    """
    Calculation of weights for the sequential importance resampling (SIR) filter.

    Inputs: xpos, ypos  - vectors with end positions from ensemble 
                          of trajectory simulations
            xobs, yobs  - observed end position of drifter
            R           - observation error variance
            
    Output: w           - weight of each particle (normalized)
    """
    
    # Calculate innovations, obs-model
    d = np.array([np.sqrt( (xobs-xpos[i])**2 + (yobs-ypos[i])**2 ) for i in range(len(xpos))])

    # Introduce temporary variable for weight calculation (not strictly necessary for our
    # simple example, but important for multiple observations and larger state space).
    v = d/R

    # Temporary variable for better numerical properties.
    wmin = np.min(-0.5*d*v)

    # Calculate weights
    w = np.exp(-0.5*d*v + wmin)

    # Normalize weights
    w = w/np.sum(w)

    return w           



def resample(w):

    """
    Resampling using SIR (van Leuween, 2009, Sec. 3a).

    Input:  w           - weights

    Output: nsamples    - number of samples of each particle
    """

    # Number of weights
    N = len(w)

    # Random number for resampling intervals
    rnum = np.random.random_sample()

    # Intervals for resampling
    rarray = (rnum + np.arange(N))/N

    # Line up weights
    warray = np.concatenate(([0],np.cumsum(w)))

    # Get samples per particle
    nsamples, rw = np.histogram(rarray, bins=warray)

    return nsamples


### Example
#np.random.seed(43)
#
#x = np.random.rand(10)
#y = np.random.rand(10)
#
#w = getWeights(x,y,0.9,0.1,0.1)
#
#r = resample(w)




























