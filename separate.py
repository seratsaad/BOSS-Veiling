import numpy as np
import glob
from astropy.io import fits
import os
from astropy.table import Table, Column
from tqdm import tqdm
import re

t = Table.read('ctts_veiling.fits')

fits_files = glob.glob('fits_files/*.fits')
t['good_wl'] = 0.

for i in tqdm(range(len(t))):
    try:
        with fits.open(fits_files[i]) as hdul:
            data = hdul[1].data
            flux = data['flux']
            model = data['model']
            wl = data['la']
        
            idx = len(wl) - 1
            while idx >= 0:
                if abs(flux[idx] - model[idx]) < 0.1:
                    break
                idx -= 1
        str = fits_files[i].split('/')
        str1 = str[1].split('.fits')
        str2 = str1[0].split('_')
        ra = float(str2[0])
        dec = float(str2[1])
        a = np.where((ra == t['RACAT']) & (dec == t['DECCAT']))[0]
        t['good_wl'][a] = wl[idx]
    except:
        continue
        

wl_arr = t['wavelength_arr'][0]

t["corrected_veiling_arr"] = Column(length=len(t),dtype=float, shape=(70,))+np.nan
t["corrected_wavelength_arr"] = Column(length=len(t),dtype=float, shape=(70,))+np.nan

for i in tqdm(range(len(t))):
    veiling_arr = t['veiling_arr'][i]
    good_wl = t['good_wl'][i]
    
    a = np.where(wl_arr < good_wl)[0]
    
    c_veiling_arr = veiling_arr[a]
    c_wavelength_arr = wl_arr[a]
    
    t["corrected_veiling_arr"][i][0:len(a)] = c_veiling_arr
    t["corrected_wavelength_arr"][i][0:len(a)] = c_wavelength_arr
    
    
t.write('ctts_veiling.fits', overwrite=True)
