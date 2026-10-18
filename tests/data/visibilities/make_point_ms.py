"""
Make point.ms, the measurement set of the tests of the conventions of
visibilities against CASA (tests/unit/test_casa_visibilities.py): CASA's
prediction of the visibilities of a point of 1 Jy, 20 arcsec east and 15
arcsec north of the phase centre (on its tangent plane), by an array of 5
ALMA antennas over 2 hours, in 4 channels from 230 GHz (LSRK), in XX and
YY; no primary beam and no noise. (CASA 6.7.3.21: the Linux wheels of
6.7.6.14, manylinux_2_28, do not load against the OpenSSL of Ubuntu
24.04.)

Run it with modular CASA (casatools and casatasks; not a dependency of
gbkfit), in this directory: python make_point_ms.py
"""

import math
import os
import shutil

import casatools


PHASE_CENTRE = (150.0, -30.0)
OFFSET = (20.0, 15.0)


def tangent_to_sky(east, north, centre):
    """
    Return the RA and Dec (radians) of offsets east and north (arcsec) on
    the tangent plane (TAN) of a centre (degrees).
    """
    east, north = (math.radians(x / 3600) for x in (east, north))
    ra0, dec0 = (math.radians(x) for x in centre)
    denominator = math.cos(dec0) - north * math.sin(dec0)
    ra = ra0 + math.atan2(east, denominator)
    dec = math.atan2(
        math.sin(dec0) + north * math.cos(dec0),
        math.hypot(east, denominator))
    return ra, dec


def main():
    for name in ('point.ms', 'point.cl'):
        shutil.rmtree(name, ignore_errors=True)
    me = casatools.measures()
    sm = casatools.simulator()
    sm.open('point.ms')
    sm.setconfig(
        telescopename='ALMA', x=[0, 300, -400, 700, -200],
        y=[0, 500, 200, -600, -800], z=[0, 0, 0, 0, 0],
        dishdiameter=[12] * 5, mount=['ALT-AZ'] * 5,
        antname=[f'A{i}' for i in range(5)],
        padname=[f'P{i}' for i in range(5)], coordsystem='local',
        referencelocation=me.observatory('ALMA'))
    sm.setspwindow(
        spwname='CO', freq='230GHz', deltafreq='10MHz',
        freqresolution='10MHz', refcode='LSRK', nchannels=4,
        stokes='XX YY')
    sm.setfeed(mode='perfect X Y', pol=[''])
    sm.setfield(
        sourcename='galaxy', sourcedirection=me.direction(
            'J2000', f'{PHASE_CENTRE[0]}deg', f'{PHASE_CENTRE[1]}deg'))
    sm.setlimits(shadowlimit=0.001, elevationlimit='8deg')
    sm.setauto(autocorrwt=0.0)
    sm.settimes(
        integrationtime='600s', usehourangle=True,
        referencetime=me.epoch('UTC', '2025/06/01/00:00:00'))
    sm.observe('galaxy', 'CO', starttime='-3600s', stoptime='3600s')
    sm.setvp(dovp=False, usedefaultvp=False)
    ra, dec = tangent_to_sky(*OFFSET, PHASE_CENTRE)
    cl = casatools.componentlist()
    cl.addcomponent(
        dir=me.direction('J2000', f'{ra}rad', f'{dec}rad'), flux=1.0,
        fluxunit='Jy', freq='230GHz', shape='point',
        spectrumtype='spectral index', index=0)
    cl.rename('point.cl')
    cl.close()
    sm.predict(complist='point.cl')
    sm.close()
    shutil.rmtree('point.cl')
    # (the predicted visibilities are in DATA; the copies are not needed)
    tb = casatools.table()
    tb.open('point.ms', nomodify=False)
    tb.removecols(['MODEL_DATA', 'CORRECTED_DATA'])
    tb.close()
    for name in os.listdir('.'):
        if name.endswith('.log'):
            os.remove(name)


if __name__ == '__main__':
    main()
