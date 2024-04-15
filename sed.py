from SEDFit.sed import SEDFit
import matplotlib.pyplot as plt
from astropy.io import fits

hdul = fits.open('ctts_veiling.fits')
data = hdul[1].data
ra = data['RACAT'][0]
dec = data['DECCAT'][0]
teff = data['pred_logteff_1'][0]
logg = data['pred_logg_1'][0]

print(ra,  dec, teff, logg)


x=SEDFit(ra, dec, 1)
x.addguesses()
x.addrange()
x.fit()
print("Distance: {} pc".format(x.getdist()))
print("AV: {} mag".format(x.getav()))
print("Radius: {} Rsun".format(x.getr()))
print("Teff: {} K".format(x.getteff()))
print("Log g: {} ".format(x.getlogg()))
print("Fe/H: {}".format(x.getfeh()))
print("Chi squared: {}".format(x.getchisq()))
ax = x.makeplot()
plt.show()
plt.savefig('fit2.png')
x.sed['model']=x.mags
x.sed