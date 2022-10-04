########################################################################
# # Team Spectacular Stellars: 'Oswald, Emma Jo', 'Kho, Jonathan Andrew', 'Andalib, Arian Alexander', 'Stone, Ashley Taylor'
# AST 304, Fall 2022
# Michigan State University
# This header (minus this line) should go at the top of all code files.
########################################################################

"""
<Description of this module goes here: what it does, how it's used.>
"""

from telnetlib import AYT
import numpy as np
from numpy.linalg import norm
# ode.py is your routine containing integrators
from ode import fEuler, rk2, rk4

# use this to identify methods
integration_methods = {
    'Euler':fEuler,
    'RK2':rk2,
    'RK4':rk4
    }
    
# energies
def kinetic_energy(v):
    """
    Returns kinetic energy per unit mass: KE(v) = 0.5 v*v.
    
    Arguments
        v (array-like)
            velocity vector
    """
    return 0.5*np.dot(v,v)

def potential_energy(x,m):
    """
    Returns potential energy per unit mass: PE(x, m) = -m/norm(r)

    Arguments
        x (array-like)
            position vector
        m (scalar)
            total mass in normalized units
    """
    # the norm function returns the length of the vector
    r = norm(x)
    return -m/r

def total_energy(z,m):
    """
    Returns energy per unit mass: E(z,m) = KE(v) + PE(x,m)

    Arguments
        z (array-like)
            position and velocity vectors with indices 0,1 being the position in the x and y plane and 2,3 being the velocity in the xy plane.
        m (scalar)
            total mass in normalized units
    """
    # to break z into position, velocity vectors, we use array slices:
    # here z[n:m] means take elements of z with indices n <= j < m
    r = z[0:2]  # start with index 0 and take two indices: 0 and 1
    v = z[2:4]  # start with index 2 and take two indices: 2 and 3

    # replace the following two lines
    KE = kinetic_energy(v)
    PE = potential_energy(r,m)
    TE = KE + PE
    return KE

def derivs(t,z,m):
    """
    Computes derivatives of position and velocity for Kepler's problem 
    
    Arguments
        t (array-like)
            time in reduced units T= 2*pi
        z (array-like)
            position and velocity vectors with indices 0,1 being the position in the x and y plane and 2,3 being the velocity in the xy plane.
        m (scalar)
            total mass in normalized units
    Returns
        numpy array dzdt with components [ dx/dt, dy/dt, dv_x/dt, dv_y/dt ]
    """
    # Fill in the following steps
    # 1. split z into position vector and velocity vector (see total_energy for example)

    r = z[0:2]  # start with index 0 and take two indices: 0 and 1
    v = z[2:4]  # start with index 2 and take two indices: 2 and 3

    # 2. compute the norm of position vector, and use it to compute the force
    norm_x = norm(r[0])
    norm_y = norm(r[1])
    # Force per unit mass = acceleration
    r_current = np.sqrt(np.dot(r))
    
    ax = -m/r_current**3*norm_x
    ay = -m/r_current**3*norm_y
    # 3. compute drdt (array [dx/dt, dy/dt])
    dxdt = norm(v[0])
    dydt = norm(v[1])
    drdt= np.array([dxdt,dydt])
    # 4. compute dvdt (array [dvx/dt, dvy/dt])
    dvxdt = ax
    dvydt = ay
    dvdt = np.array([dvxdt, dvydt])
    # join the arrays
    dzdt = np.concatenate((drdt,dvdt))
    return dzdt

def integrate_orbit(z0,m,tend,h,method='RK4'):
    """
    Integrates orbit starting from an initial position and velocity from t = 0 
    to t = tend.
    
    Arguments:
        z0
            < the first value is the z(t) array >
        m
            < the mass >
    
        tend
            < the end time being integrated over >
    
        h
            < the step size >
    
        method ('RK4')
            identifies which stepper routine to use (default: 'RK4')
    Returns
        ts, Xs, Ys, KEs, PEs, TEs := arrays of time, x postions, y positions, 
        and energies (kin., pot., total) 
    """

    # set the initial time and phase space array
    t = 0.0
    z = z0

    # expected number of steps
    Nsteps = int(tend/h)+1

    # arrays holding t, x, y, kinetic energy, potential energy, and total energy
    ts = np.zeros(Nsteps)
    Xs = np.zeros(Nsteps)
    Ys = np.zeros(Nsteps)
    KEs = np.zeros(Nsteps)
    PEs = np.zeros(Nsteps)
    TEs = np.zeros(Nsteps)

    # store the initial point
    ts[0] = t
    Xs[0] = z[0]
    Ys[0] = z[1]
    
    # now extend this with KEs[0], PEs[0], TEs[0]
    KEs[0] = kinetic_energy(v[2:4])
    PEs[0] = potential_energy(r[0:2],m)
    TEs[0] = total_energy(z,m)

    # select the stepping method
    advance_one_step = integration_methods[method]
    # run through the steps
    for step in range(1,Nsteps):
        z = advance_one_step(derivs,t,z,h,args=m)
        # insert statement here to increment t by the stepsize h
        t = t+h
        # store values
        ts[step] = t
        # fill in with assignments for Xs, Ys, KEs, PEs, TEs
        Xs[step] = z[0]
        Ys[step] = z[1]
        KEs[step] = kinetic_energy(v[2:4])
        PEs[step] = potential_energy(r[0:2],m)
        TEs[step] = total_energy(z,m)
    return ts, Xs, Ys, KEs, PEs, TEs
    
def set_initial_conditions(a, m, e):
    
    """
    set the initial conditions for the orbit.  The orientation of the orbit is 
    chosen so that y0 = 0.0 and vx0 = 0.0.
    
    Arguments
        a
            semi-major axis in AU
        m
            total mass in Msun
        e
            eccentricity ( x0 = (1+e)*a )
    
    Returns:
    x0, y0, vx0, vy0, eps0, Tperiod := initial position and velocity, energy, 
        and period
    """
        
    # fill in the following lines with the correct formulae    
    # total energy per unit mass
    eps0 = -m/(2*a)
    
    # period of motion
    Tperiod = ((np.pi)/(np.sqrt(2))*m*abs(eps)**(-3/2)

    # initial position
    # fill in the following lines with the correct formulae
    x0 = a*(1+e) 
    y0 = 0.0

    # initial velocity is in y-direction; we compute it from the energy
    # fill in the following lines with the correct formulae
    vx0 = 0.0
    vy0 = np.sqrt(2*eps0 + (2*m/x0))
    
    return np.array([x0,y0,vx0,vy0]), eps0, Tperiod