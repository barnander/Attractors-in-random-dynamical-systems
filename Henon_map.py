#%% Packages
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#%% Set up henon map with random bounded noise
def Henon(x,a = 1.4,b = 0.3, e = 0.6):
    """
    Computes the Henon map with bounded noise.
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
    angle = random.rand()*2*np.pi #generate random angle in radians
    #to ensure uniform distribution over disk, radius is square root 
    #of random number uniformly distributed in [0,1] scaled by e
    radius = np.sqrt(random.rand())*e 
    bounded_noise = np.array([radius*np.cos(angle),radius*np.sin(angle)], ndmin = 2).T
    
    return np.array([1 - a*x1**2 + x2, b*x1],ndmin=2).T + bounded_noise
    
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

def animHenon(x0, n, a = 0.06, b = 0.3, b_val = 1, sig = 1, mu = 0):
    """
    Animates the Henon map with bounded noise.
    Parameters:
        x0 (array): 2D array of the initial state of the system.
        n (int): Number of iterations.
        a (float): Parameter of the Henon map.
        b (float): Parameter of the Henon map.
        b_val (float): Bound of the noise.
        sig (float): Standard deviation of the noise.
        mu (float): Mean of the noise.
    """
    fig, ax = plt.subplots()
    x = x0
    x_vals = np.zeros((2,n))
    for i in range(n):
        x_vals[:,i] = x
        x = Henon(x,a,b,b_val,sig,mu)
    x_vals = x_vals.T
    x_vals = np.array(x_vals)
    x_vals = x_vals.reshape(n,2)
    x_vals = x_vals.T
    line, = ax.plot(x_vals[0,:],x_vals[1,:],'o')
    def animate(i):
        line.set_data(x_vals[0,:i],x_vals[1,:i])
        return line,
    ani = animation.FuncAnimation(fig, animate, frames=n, interval=50, blit=True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Henon map with bounded noise')
    plt.show()

#%%
x0 = np.array([0.1,0.1])
n = 100
plotHenon(x0,n,sig = 0.005)
# %%
animHenon(x0,n,sig = 0.005)
# %%
