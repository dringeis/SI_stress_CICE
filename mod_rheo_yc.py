#! /usr/bin/python

import numpy as np
import xarray as xr
import pylab as plt
from matplotlib.colors import LogNorm

def open_data(fname):
    ds = xr.open_dataset(fname)
    return ds

def comp_press(ds, fname, const):
    h = ds.hi
    A = ds.aice
    Pstar = const[fname]['Pstar']
    Cstar = const[fname]['Cstar']
    P0 = h*Pstar*np.exp(-Cstar*(1-A))
    return P0

def comp_stress_inv(ds, fname, const):
    #Load constants
    e = const[fname]['e']
    kt = const[fname]['kt']
    Zmax = const[fname]['Zmax']
    Dmin = const[fname]['Dmin']

    # Load data
    P0 = comp_press(ds, fname, const)
    eI = ds.divu
    eII = ds.shear

    # Compute viscosities
    D = np.sqrt(eI**2+1/e**2*eII**2)
    zeta = P0*(1+kt)/(2*np.sqrt(D**2+Dmin**2))
    # zeta = np.minimum(zeta, Zmax)
    eta = zeta/e**2

    # Compute stresses
    sI = zeta*eI - P0*(1-kt)/2
    sII = eta*eII

    return sI, sII

def norm_stress_inv(ds, fname, const):
    P0 = comp_press(ds, fname, const)
    sI, sII = comp_stress_inv(ds, fname, const)
    return sI/P0, sII/P0

def ell_th(fname, const, n):
    kt = const[fname]['kt']
    e = const[fname]['e']
    dx = (1+kt)/n
    sIth = np.arange(-1, kt+dx, dx)
    sIIth = 1./e*np.sqrt(kt-sIth**2-sIth*(1-kt))
    return sIth, sIIth

def plot_yc(ds, fname, const):
    sIn, sIIn = norm_stress_inv(ds, fname, const)
    e = const[fname]['e']
    kt = const[fname]['kt']
    eI = ds.divu
    eII = ds.shear

    sIth, sIIth = ell_th(fname, const, 100)

    s=10
    fig, ax = plt.subplots()
    ax.plot(sIth, sIIth, '-r', lw=1)
    ax.plot(sIn[0], sIIn[0], 'b.', ms=0.5)
    ax.quiver(sIn[0][::s,::s],sIIn[0][::s,::s],eI[0][::s,::s],eII[0][::s,::s])
    ax.grid()
    ax.axis('scaled')

def plot_hist2D(ds, fname, const):
    sIn, sIIn = norm_stress_inv(ds, fname, const)
    e = const[fname]['e']

    sIth, sIIth = ell_th(fname, const, 100)

    sIi = sIn[0].values.flatten()
    sIIi = sIIn[0].values.flatten()

    bad = np.isnan(sIi) | np.isnan(sIIi)
    good = ~bad
    sIp = sIi[good]
    sIIp = sIIi[good]

    fig, ax = plt.subplots()
    ax.plot(sIth, sIIth, '-k', lw=2)
    h2d = ax.hist2d(sIp, sIIp, bins=100, cmap='Reds',density=True, norm=LogNorm())
    plt.colorbar(h2d[3],ax=ax)
    ax.grid()
    ax.axis('scaled')


if __name__=="__main__":

    fname1 = '2005032815_000.nc'

    #constants dictionnary
    const={}
    const[fname1]={'Pstar':2.75e4, 'e':2.0, 'Cstar':15., 'Dmin':1e-4, 'Zmaxf':2.5e4, 'kt':0.05}
    const[fname1]['Zmax'] = const[fname1]['Zmaxf']*const[fname1]['Pstar']

    ds = open_data(fname1)

    plot_yc(ds, fname1, const)

    plot_hist2D(ds, fname1, const)

    plt.show()


'''
2005032815_000.nc
<xarray.Dataset>
Dimensions:  (time: 1, nj: 2198, ni: 1580)
Coordinates:
  * time     (time) object 2011-03-23 15:00:00
    TLON     (nj, ni) float32 ...
    TLAT     (nj, ni) float32 ...
    ULON     (nj, ni) float32 ...
    ULAT     (nj, ni) float32 ...
Dimensions without coordinates: nj, ni
Data variables:
    hi       (time, nj, ni) float32 ...
    aice     (time, nj, ni) float32 ...
    uvel     (time, nj, ni) float32 ...
    vvel     (time, nj, ni) float32 ...
    uocn     (time, nj, ni) float32 ...
    vocn     (time, nj, ni) float32 ...
    strairx  (time, nj, ni) float32 ...
    strairy  (time, nj, ni) float32 ...
    strocnx  (time, nj, ni) float32 ...
    strocny  (time, nj, ni) float32 ...
    strintx  (time, nj, ni) float32 ...
    strinty  (time, nj, ni) float32 ...
    divu     (time, nj, ni) float32 ...
    shear    (time, nj, ni) float32 ...
    windx    (time, nj, ni) float32 ...
    windy    (time, nj, ni) float32 ...
    tau_bu   (time, nj, ni) float32 ...
    tau_bv   (time, nj, ni) float32 ...
    uvelU    (time, nj, ni) float32 ...
    vvelU    (time, nj, ni) float32 ...
Attributes:
    title:        sea ice model output for CICE
    contents:     Diagnostic and Prognostic Variables
    source:       sea ice model: Community Ice Code (CICE)
    comment:      All years have exactly 365 days
    comment2:     File written on model date 20050328
    comment3:     seconds elapsed into model date:  54000
    conventions:  CF-1.0
    history:      This dataset was created on 2022-02-24 at 03:48:31.9
'''
