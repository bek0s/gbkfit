
import astropy.io.fits
import astropy.wcs


def load_fits(filename):
    data = astropy.io.fits.getdata(filename)
    wcs = astropy.wcs.WCS(astropy.io.fits.getheader(filename))
    return data, wcs


def dump_fits(filename, data, wcs, overwrite=False):
    astropy.io.fits.writeto(
        filename, data, header=wcs.to_header(),
        output_verify='exception', overwrite=overwrite, checksum=True)
