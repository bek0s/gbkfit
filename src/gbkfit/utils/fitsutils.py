
import astropy.io.fits
import astropy.wcs
import numpy as np


def load_fits(
        filename: str,
        hdu: int = 0,
        memmap: bool = True
) -> tuple[np.ndarray, astropy.wcs.WCS]:
    """Loads a FITS file and extracts data and WCS."""
    with astropy.io.fits.open(filename, memmap=memmap) as hdulist:
        data = hdulist[hdu].data
        header = hdulist[hdu].header
        wcs = astropy.wcs.WCS(header)
    return data, wcs


def dump_fits(
        filename: str,
        data: np.ndarray,
        wcs: astropy.wcs.WCS | None = None,
        overwrite: bool = False
) -> None:
    """Saves data to a FITS file, including WCS if available."""
    header = wcs.to_header() if wcs else None
    astropy.io.fits.writeto(
        filename, data, header=header,
        output_verify='exception', overwrite=overwrite, checksum=True)
