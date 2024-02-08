# S. Saad


# IMPORT BLOCK
###############################
###############################

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


# FUNCTIONS
###############################
###############################
def reading_obs(file_path, rv):
    hdul = fits.open(file_path)
    t = hdul[1].data
    a = np.where((t["IVAR"] > np.median(t["IVAR"])/10) & (np.isfinite(t['IVAR'])==True))[0]
    flux = t["FLUX"][a] / np.median(t["FLUX"][a])
    wl2 = 10**(t["LOGLAM"][a])
    wl2 = wl2 + wl2*(rv/299792.458)
    err=np.sqrt(1./t["IVAR"][a])
    #a = np.where((wl2<x) & (wl2>=x))[0]
    #flux = flux[a]
    #wl2 = wl2[a]
    return flux, wl2,err


def divide_in_steps(flux, wl2):
    total_steps = int(((max(wl2)-min(wl2))//100))
    wl2_arr = []
    flux_arr = []
    for i in range(total_steps):
        x = ((min(wl2)//100)*100) + 100*i
        a = np.where((wl2<x+1000) & (wl2>=x))[0]
        flux_new = flux[a]
        wl2_new = wl2[a]
        flux_arr.append(flux_new)
        wl2_arr.append(wl2_new)
    return flux_arr, wl2_arr

def reading_ph(teff, logg):
    hdul = fits.open(f"C:/Users/serat/Downloads/phoenix_fits/lte0{teff}-{logg}0-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    primary_hdu = hdul[0]
    data = primary_hdu.data
    header = primary_hdu.header
    wl1 = np.arange(header["NAXIS1"])*header["CDELT1"]+header["CRVAL1"]
    data = data/np.median(data)
    return data, wl1

def model_flux_Av(synthetic_flux, Av):
    ext_model = G23(Rv=3.1)
    flux_corrected = synthetic_flux * ext_model.extinguish(wavelengths_obs, Av=Av)
    flux = flux_corrected/np.median(flux_corrected)
    return flux
    
def model_flux_veiling(flux, v_const):
    flux = (flux/np.median(flux)) + v_const
    flux = flux/(1+v_const)
    return flux/np.median(flux)

def reinterp(wavelengths_obs,wavelengths_ph, flux_ph):
    ll=np.diff(wavelengths_obs)
    d=[ll[0]]
    d.extend(ll)
    d.append(ll[-1])
    d=np.array(d)
    l1=wavelengths_obs-d[:-1]*0.5
    l2=wavelengths_obs+d[1:]*0.5
    
    synthetic_flux=np.zeros(len(l1))
    for j in range(len(l1)):
        x=np.where((wavelengths_ph>l1[j]) & (wavelengths_ph<l2[j]))[0]
        synthetic_flux[j]=np.sum(flux_ph[x])/len(x)

    return synthetic_flux
    




# MAIN
###############################
###############################
s = Table.read("lineforest_boss.fits")

teff = round(10**(s["u_med_logteff_1"][0])/100)*100
logg = round(s["u_med_logg_1"][0] / 0.5)*0.5


_, wavelengths_ph = reading_ph(teff, logg) # the common wavelength range of the phoenix data


ext_model = G23(Rv=3.1)


# Saving all the possible Teff and Logg in the phoenix data
a = glob.glob("C:/Users/serat/Downloads/phoenix_fits/lte0*-*0-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
Teff = []
Logg = []

for i in range(len(a)):
    file = a[i]
    splt = file.split("lte")
    teff = int(splt[1].split("-")[0])
    logg = float(splt[1].split("-")[1])
    Teff.append(teff)
    Logg.append(logg)

Teff = np.unique(Teff)
Logg = np.unique(Logg)


# array used to interpolate
arr = np.zeros((len(Teff), len(Logg), len(wavelengths_ph)),)


for i in range(len(Teff)):
    for j in range(len(Logg)):
        try:
            flux1, wl1 = reading_ph(Teff[i], Logg[j])
            arr[i][j] = flux1
        except:
            continue
            
interpolator_ph = RegularGridInterpolator((Teff, Logg), arr)


s["Av"] = 0.
s["Chisquare_list"] = Column(length=len(s),dtype=float, shape=(4,))+np.nan
s["veiling_arr"] = Column(length=len(s),dtype=float, shape=(70,))+np.nan

arr = np.arange(0, 67, 1)

for i in tqdm(range(100)):
    try:
        Teff_obs = 10**(s["u_med_logteff"][i])
        Logg_obs = s["u_med_logg"][i]
        healpix = s["HEALPIX_PATH"][i]
        file_path = f"/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/healpix{healpix[12:]}"
        rv = s["XCSAO_RV"][i]
    
        flux_obs, wavelengths_obs, err_obs = reading_obs(file_path, rv)
        
        flux_ph = interpolator_ph((Teff_obs, Logg_obs))
        synthetic_flux = reinterp(wavelengths_obs, wavelengths_ph, flux_ph)
        
        wavelengths_obs = wavelengths_obs * u.AA
        wavenumbers = (1.0 / wavelengths_obs)
        a = np.where((wavenumbers >= 0.03125/u.micron) & (wavenumbers <= 10.964912280701753/u.micron) & (wavenumbers != np.nan))[0]
        synthetic_flux = synthetic_flux[a]
        wavelengths_obs = wavelengths_obs[a]
        flux_obs = flux_obs[a]
        err_obs=err_obs[a]    
            
        # Normalizing everything again
        err_obs=err_obs//np.median(flux_obs)
        flux_obs = flux_obs/np.median(flux_obs)
        synthetic_flux = synthetic_flux/np.nanmedian(synthetic_flux)
        b = np.where(np.isfinite(synthetic_flux) == True)[0]
        
        synthetic_flux=np.interp(wavelengths_obs,wavelengths_obs[b],synthetic_flux[b])
        
        params, _ = curve_fit(model_flux_Av, synthetic_flux, flux_obs, bounds=(0, 100), sigma=err_obs)
        
        s['Av'][i] = params[0]
        
        flux_correct = synthetic_flux*ext_model.extinguish(wavelengths_obs, Av=params[0])
        flux_correct = flux_correct/np.median(flux_correct)
        wavelengths_obs = wavelengths_obs/u.AA
        
        flux_correct_arr, wl_arr = divide_in_steps(flux_correct, wavelengths_obs)
        flux_obs_arr, wl_arr = divide_in_steps(flux_obs, wavelengths_obs)
        err_obs_arr, wl_arr = divide_in_steps(err_obs, wavelengths_obs)
        
        veiling_arr = []
       
            
        for j in range(len(wl_arr)):
            synthetic_flux_step = flux_correct_arr[j]/np.median(flux_correct_arr[j])
        
            flux_obs_step=flux_obs_arr[j]/np.median(flux_obs_arr[j])
            flux_obs_arr[j]=flux_obs_step
            
            err_obs_step=err_obs_arr[j]/np.median(flux_obs_step)
            err_obs_arr[j]=err_obs_step
            
            wl_step = wl_arr[j]
            
            params, _ = curve_fit(model_flux_veiling, synthetic_flux_step, flux_obs_step, bounds=(0, 20), sigma=err_obs_step)
            flux_correct_arr[j] = (synthetic_flux_step + params[0]) / (1+params[0])
            veiling_arr.append(params[0])
            
        # chi square for each range
        
        wl_arr_comb = np.concatenate(wl_arr)
        err_obs_comb = np.concatenate(err_obs_arr)
        flux_correct_comb = np.concatenate(flux_correct_arr)
        flux_obs_comb = np.concatenate(flux_obs_arr)
        
        range1_mask = np.where((wl_arr_comb > 5500) & (wl_arr_comb < 6500))
        range2_mask = np.where((wl_arr_comb > 6600) & (wl_arr_comb < 7600))
        range3_mask = np.where((wl_arr_comb > 7600) & (wl_arr_comb < 8600))
        range4_mask = np.where((wl_arr_comb > 8600) & (wl_arr_comb < 9600))
        
        
        chi_square_range1 = np.sum(((flux_obs_comb[range1_mask] - flux_correct_comb[range1_mask]) / err_obs_comb[range1_mask]) ** 2)
        chi_square_range2 = np.sum(((flux_obs_comb[range2_mask] - flux_correct_comb[range2_mask]) / err_obs_comb[range2_mask]) ** 2)
        chi_square_range3 = np.sum(((flux_obs_comb[range3_mask] - flux_correct_comb[range3_mask]) / err_obs_comb[range3_mask]) ** 2)
        chi_square_range4 = np.sum(((flux_obs_comb[range4_mask] - flux_correct_comb[range4_mask]) / err_obs_comb[range4_mask]) ** 2)

        s["Chisquare_list"][i][0] = chi_square_range1
        s["Chisquare_list"][i][1] = chi_square_range2
        s["Chisquare_list"][i][2] = chi_square_range3
        s["Chisquare_list"][i][3] = chi_square_range4
        
            
        s["veiling_arr"][i][0 : len(veiling_arr)] = veiling_arr
            
    

    except:
        print(f"Dead index: {i}")
        continue

    
s.write("lineforest_veiling.fits", overwrite=True)