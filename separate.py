import numpy as np
import glob
from astropy.io import fits
import os
from astropy.table import Table

t = Table.read('ctts_veiling.fits')

fits_files = glob.glob('fits_files/*.fits')


for i in range(10):
    with fits.open(fits_files[i]) as hdul:
        data = hdul[1].data
        flux = data['flux']
        model = data['model']
        wl = data['la']
        
        idx = np.where(abs(flux - model) < 0.4)[0]
        print(wl[idx[0]], wl[idx[-1]])
        
        #t['good_wl_min'] = wl[idx[0]]
        #t['good_wl_max'] = wl[idx[-1]]
        
#t.write('ctts_veiling.fits', overwrite-True)
    
        
