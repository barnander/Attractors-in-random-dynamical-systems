#%% Packages
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#%% Set up henon map with random bounded noise
def Henon(x,a = 0.607,b = 0.3, e = 0.0625):
    """
    Computes the Henon map with additive, spherical bounded noise.
    Parameters:
        x (array): 2D array of the current state of the system.
        a (float): Parameter of the Henon map.
        b (float): Parameter of the Henon map.
        b_val (float): Bound of the noise.
        sig (float): Standard deviation of the noise.
        mu (float): Mean of the noise.
    """
    # Seperate out stae variables
    x1 = x[0,:]
    x2 = x[1,:]
    #Generate noise uniformly distributed noise over disk of radius e from the origin
    dim = x.shape[1]
    angle = random.rand(dim)*2*np.pi #generate random angle in radians
    #to ensure uniform distribution over disk, radius is square root 
    #of random number uniformly distributed in [0,1] scaled by e
    radius = np.sqrt(random.rand(dim))*e 
    bounded_noise = np.array([radius*np.cos(angle),radius*np.sin(angle)], ndmin = 2)
    #Compute the Henon map with bounded noise
    detSys = np.array([1 - a*x1**2 + x2, b*x1],ndmin=2)
    return np.array([1 - a*x1**2 + x2, b*x1],ndmin=2) + bounded_noise
    
#%% plot the Henon map
def plotHenon(x0, n, a = 0.06, b = 0.3, b_val = 1, sig = 1, mu = 0):
    """
    Plots the Henon map with bounded noise.
    Parameters:
        x0 (array): 2D array of the initial state of the system.
        n (int): Number of iterations.
        a (float): Parameter of the Henon map.
        b (float): Parameter of the Henon map.
        b_val (float): Bound of the noise.
        sig (float): Standard deviation of the noise.
        mu (float): Mean of the noise.
    """
    x = x0
    x_vals = np.zeros((2,n))
    for i in range(n):
        x_vals[:,i] = x
        x = Henon(x,a,b,b_val,sig,mu)
    plt.plot(x_vals[0,:],x_vals[1,:],'o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Henon map with bounded noise')
    plt.show()


#%%
#checking random noise distribution
e = 0.0625
n = 1000
angle = random.rand(1,n)*2*np.pi #generate random angle in radians
    #to ensure uniform distribution over disk, radius is square root 
    #of random number uniformly distributed in [0,1] scaled by e
radius = np.sqrt(random.rand(1,n))*e 
bounded_noise = np.vstack((radius*np.cos(angle),radius*np.sin(angle)))
    
#plot circle radius e
theta = np.linspace(0,2*np.pi,100)
x = e*np.cos(theta)
y = e*np.sin(theta)
plt.plot(x,y)
#plot bounded noise
plt.scatter(bounded_noise[0,:],bounded_noise[1,:])
plt.show()
# %% plot steady minimal attractor of Henon map 
# trying to reproduce plot in slides
# multiple starting points to reduce for loops
nStart = 50000
x0s = random.uniform(0,1,(2,nStart))

n = 10000 #number of iterations
a = 0.607
b = 0.3
e = 0.0625
x = x0s
for i in range(n):
    x = Henon(x,a,b,e)
    #find values that have diverged
    notDiverged = np.linalg.norm(x,axis = 0) < 100
    #count them
    noFinite = np.sum(notDiverged)
    # remove the initial points that have diverged 
    # from bassin of attraction set
    x0s = x0s[:,notDiverged]
    #take only values that haven't diverged
    x = x[:,notDiverged]
    """
    #add new random starting points to replace diverged ones
    x0New = random.uniform(-1,2,(2,nStart- noFinite))
    x0s = np.hstack((x0s,x0New))
    x = np.hstack((x,x0New))
    """
    print(i)
plt.figure()
plt.scatter(x[0,:],x[1,:],marker='x', s=0.5)
plt.scatter(x0s[0,:],x0s[1,:],marker='x', s=0.5, c = 'red')
#plt.xlim(-1, 1.75)
#plt.ylim(-0.8, 0.8)
plt.show()
# %%
