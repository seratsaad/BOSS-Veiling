from SEDFit.sed import SEDFit
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from astropy.table import Table



hdul = fits.open('ctts_veiling.fits')
data = hdul[1].data

print(len(data))

for i in tqdm(range(len(data))):
    try:
        ra = data['RACAT'][i]
        dec = data['DECCAT'][i]
        teff = data['pred_logteff_1'][i]
        logg = data['pred_logg_1'][i]

        x=SEDFit(ra, dec, 1, download_gaia=True)
        x.addguesses(teff=10**teff, logg=logg,feh=0)
        x.sed['model']=x.mags
        table = x.sed
        print(table)
        table.write(f'fits_files/{ra}_{dec}.fits', format='fits', overwrite=True)
    except:
        continue


