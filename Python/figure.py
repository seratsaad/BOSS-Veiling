import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from tqdm import tqdm
import math
from astropy.io import fits
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator
from dust_extinction.parameter_averages import F99
from dust_extinction.parameter_averages import G23
from scipy.signal import savgol_filter

from scipy.optimize import curve_fit
import matplotlib.cm as cm
import sys

def reading_obs(file_path, rv):
    hdul = fits.open(file_path)
    t = hdul[1].data
    a = np.where((t["IVAR"] > np.median(t["IVAR"])/10) & (np.isfinite(t['IVAR'])==True))[0]
    flux = t["FLUX"][a]
    wl2 = 10**(t["LOGLAM"][a])
    wl2 = wl2 + wl2*(rv/299792.458)
    err = (np.sqrt(1./t["IVAR"][a])) / np.nanmedian(flux)

    return flux, wl2, err

def reading_ph(teff, logg):
    hdul = fits.open(f"phoenix_fits/lte0{teff}-{logg}0-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    primary_hdu = hdul[0]
    data = primary_hdu.data
    header = primary_hdu.header
    wl1 = np.arange(header["NAXIS1"])*header["CDELT1"]+header["CRVAL1"]
    data = data/np.median(data)
    return data, wl1

def main():
    s = Table.read('boss_ysos_veiling19.fits')
    for i in tqdm(range(3,4)):
        rv = s["XCSAO_RV"][i]
        teff_obs = round(10**(s["u_med_logteff_1"][i])/100)*100
        logg_obs = round(s["u_med_logg_1"][i] / 0.5)*0.5
        healpix = s["HEALPIX_PATH"][i]
        file_path = f"/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/healpix{healpix[12:]}"

        flux_obs, wavelengths_obs, err_obs = reading_obs(file_path, rv)
        
        Av = s['av'][i]
        flux_corrected = flux_obs * 10**(0.4*Av)

        flux_ph, wl_ph = reading_ph(teff_obs, logg_obs)

        plt.rc('text', usetex=True)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 18
        })

        median_veil = np.nanmean(s['veiling_arr'][i])
        flux_veil_corr = (flux_corrected + median_veil) / (1+median_veil)
        
        #flux_obs_norm = flux_obs / np.median(flux_obs)
        #flux_corrected_norm = flux_corrected / np.median(flux_corrected)
        #flux_veil_corr_norm = flux_veil_corr / np.median(flux_veil_corr)
        #flux_ph_norm = flux_ph / np.median(flux_ph)
        
        plt.clf()
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(wavelengths_obs, flux_obs, label='Observed Spectra')
        axs[0].plot(wavelengths_obs, flux_corrected, label='Extinction Corrected Spectra')
        axs[0].plot(wavelengths_obs, flux_veil_corr, label='Veiling Corrected Spectra')
        axs[0].set_xlim(4000, 10000)
        axs[0].set_ylim(-0.1, 1500)
        axs[0].set_ylabel('Flux')
        #axs[0].legend()

        axs[1].plot(wl_ph, flux_ph, label='Phoenix Model')
        axs[1].set_xlim(4000, 10000)
        axs[1].set_ylim(-0.01, 1.6)
        axs[1].set_xlabel('Wavelength (\AA)')
        axs[1].set_ylabel('Normalized Flux')
        #axs[1].legend()

        plt.tight_layout()
        plt.savefig(f'figures/spectra_correction.pdf')
        #plt.savefig(f'figures/observed_corrected_phoenix_{i}.png')
        plt.show()

if __name__ == "__main__":
    main()
