"""
Tests for the API of the server: its token, and the round trips of a
project, a job and its run, and a live session.
"""

import json
import os
import pathlib
import time

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from gbkfit_server.api import create_app, openapi
from gbkfit_server.services import frames
from gbkfit_server.services.errors import Invalid
from gbkfit_server.services.executors import LocalExecutor
from gbkfit_server.services.server import Server
from gbkfit_server.services.workspace import Workspace


TOKEN = 'secret'

CONTRACT = pathlib.Path(__file__).parents[1] / 'openapi.json'


@pytest.fixture
def client(root):
    """A client of a server of the workspace, with the token."""
    app = create_app(Server(root, LocalExecutor()), TOKEN, {})
    with TestClient(app, headers=dict(authorization=f'Bearer {TOKEN}')) \
            as client:
        yield client


def test_the_api_needs_the_token(client):
    assert client.get('/api/v1/projects').json() == ['study']
    assert client.get(
        '/api/v1/projects', headers=dict(authorization='Bearer wrong')
    ).status_code == 401
    assert client.get(
        '/api/v1/projects', headers=dict(authorization=''),
        params=dict(token=TOKEN)).status_code == 200
    # (the information of the server does not)
    assert client.get('/api', headers=dict(authorization='')).json()['api'] \
        == '/api/v1'


def test_a_job_writes_a_run_whose_arrays_are_read(client, setup):
    response = client.post('/api/v1/jobs', json=dict(task='eval', options=dict(
        mode='model', config=setup, profile_iters=0)))
    assert response.status_code == 201
    job = response.json()
    for _ in range(600):
        job = client.get(f"/api/v1/jobs/{job['id']}").json()
        if job['state'] not in ('queued', 'running'):
            break
        time.sleep(0.1)
    assert job['state'] == 'complete', job['output']
    run = client.get(f"/api/v1/runs/{job['run']}").json()
    (model,) = [p for p in run['products'] if p['kind'] == 'model']
    frame = client.get(f"/api/v1/arrays/{job['run']}/{model['path']}").content
    (image,) = frames.arrays(*frames.decode(frame))
    assert image.shape == (12, 16)
    assert client.get('/api/v1/runs/study/nothing').status_code == 404
    # (any file of it as it is, e.g. its configuration)
    raw = client.get(f"/api/v1/raw/{job['run']}/config.yaml")
    assert raw.status_code == 200 and b'galaxies' in raw.content
    assert client.get('/api/v1/raw/study/nothing.png').status_code == 404


def test_a_session_evaluates_values_sent_over_a_websocket(client, setup):
    session = client.post('/api/v1/sessions', json=dict(config=setup)).json()
    assert session['free']['incl']['value'] == 45
    url = f"/api/v1/sessions/{session['id']}/evaluate?token={TOKEN}"
    with client.websocket_connect(url) as websocket:
        # (the frame of the data comes first: here, none)
        assert frames.decode(websocket.receive_bytes())[0]['type'] == 'ready'
        websocket.send_json(dict(values=dict(incl=80)))
        header, data = frames.decode(websocket.receive_bytes())
    assert header['type'] == 'result' and header['chi2'] is None
    assert frames.arrays(header, data)[0].shape == (12, 16)
    assert client.delete(
        f"/api/v1/sessions/{session['id']}").status_code == 204


def test_the_contract_is_that_of_the_api():
    # (python -m gbkfit_server.api > openapi.json writes it)
    assert json.loads(CONTRACT.read_text()) == openapi()


def test_the_pages_are_served_and_the_link_leads_to_the_first(root, tmp_path):
    page = tmp_path / 'page'
    page.mkdir()
    (page / 'index.html').write_text('<title>Studio</title>')
    app = create_app(Server(root, LocalExecutor()), TOKEN, dict(studio=page))
    with TestClient(app) as client:
        response = client.get(f'/?token={TOKEN}', follow_redirects=False)
        assert response.headers['location'] == f'/studio/?token={TOKEN}'
        assert '<title>Studio</title>' in client.get('/studio/').text
        assert client.get('/api').json()['pages'] == ['studio']


def test_a_text_is_checked_in_place(client, root, setup):
    # A setup with an unknown section (an error), and a piece with a
    # misspelt type (in the setup that names it)
    text = (pathlib.Path(root) / setup).read_text() + 'bogus: 1\n'
    (diagnostic,) = client.post('/api/v1/check', json=dict(
        path=setup, text=text)).json()['diagnostics']
    assert diagnostic['severity'] == 'error'
    assert "'bogus'" in diagnostic['message']
    piece = 'study/galaxies/disk.yaml'
    text = (pathlib.Path(root) / piece).read_text().replace(
        'exponential', 'exponentail')
    (diagnostic,) = client.post('/api/v1/check', json=dict(
        path=piece, text=text)).json()['diagnostics']
    assert diagnostic['severity'] == 'error'
    assert "did you mean 'exponential'" in diagnostic['message']
    assert diagnostic['message'].endswith('(in setups/disk.yaml)')
    lines = text.splitlines()
    assert 'ptraits' in lines[diagnostic['line'] - 1]


def test_projects_and_pieces_are_created_from_templates(client):
    response = client.post('/api/v1/projects', json=dict(path='new'))
    assert response.status_code == 201
    assert response.json()['pieces']['setups'] == []
    assert client.post('/api/v1/projects', json=dict(
        path='new/inner')).status_code == 400
    path = client.post('/api/v1/pieces', json=dict(
        project='new', folder='galaxies', name='disk')).json()
    assert path == 'new/galaxies/disk.yaml'
    assert 'type: disk' in client.get(f'/api/v1/files/{path}').text
    assert client.post('/api/v1/pieces', json=dict(
        project='new', folder='galaxies', name='disk')).status_code == 400
    assert sorted(client.get('/api/v1/projects').json()) == ['new', 'study']


def test_pieces_are_created_with_their_content_and_read(client):
    content = dict(type='pixel_spectra', data=['data/cube.fits', 'SCI'])
    path = client.post('/api/v1/pieces', json=dict(
        project='study', folder='datasets', name='kmos', content=content)
    ).json()
    assert path == 'study/datasets/kmos.yaml'
    assert client.get(f'/api/v1/pieces/{path}').json() == content
    assert client.get('/api/v1/pieces/study/datasets/no.yaml').status_code \
        == 404


def test_pieces_are_renamed_with_their_names_and_deleted(client):
    assert client.get('/api/v1/references/study/observations/image.yaml') \
        .json() == [dict(folder='setups', name='disk')]
    response = client.patch(
        '/api/v1/pieces/study/observations/image.yaml',
        json=dict(name='sky/image'))
    assert response.json() == 'study/observations/sky/image.yaml'
    assert client.get('/api/v1/pieces/study/setups/disk.yaml').json()[
        'observations'] == 'sky/image'
    assert client.patch(
        '/api/v1/pieces/study/galaxies/disk.yaml', json=dict(name='../x')
    ).status_code == 400
    assert client.patch(
        '/api/v1/pieces/study/data/disk.yaml', json=dict(name='x')
    ).status_code == 404
    assert client.delete(
        '/api/v1/pieces/study/observations/sky/image.yaml').status_code == 204
    assert client.get('/api/v1/projects/study').json()['pieces'][
        'observations'] == []
    assert client.delete(
        '/api/v1/pieces/study/observations/sky/image.yaml').status_code == 404


def test_pieces_are_updated_and_their_comments_stay(client):
    path = client.post('/api/v1/pieces', json=dict(
        project='study', folder='psfs', name='beam')).json()
    content = dict(type='moffat', alpha=1.2, beta=3)
    assert client.put(f'/api/v1/pieces/{path}', json=content).status_code \
        == 204
    assert client.get(f'/api/v1/pieces/{path}').json() == content
    assert client.get(f'/api/v1/files/{path}').text.startswith('# A PSF')
    assert client.put('/api/v1/pieces/study/psfs/no.yaml', json=content) \
        .status_code == 404


def test_psfs_and_lsfs_are_drawn_with_their_widths(client, root):
    psf = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(type='gauss', sigma=1.0))).json()
    values = np.array(psf['values'])
    assert values.shape[0] == values.shape[1] and values.shape[0] > 100
    assert np.isclose(values.sum(), 1, atol=1e-4)
    assert np.isclose(psf['fwhm'], 2 * np.sqrt(2 * np.log(2)), rtol=1e-6)
    # (on the pixels of data)
    psf = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(type='gauss', sigma=1.0),
        step=[0.5, 0.5])).json()
    assert psf['step'] == [0.5, 0.5] and len(psf['values']) < 100
    point = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(type='point'))).json()
    assert point['values'] == [[1.0]] and point['fwhm'] is None
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='gauss', sigma=10))).json()
    assert np.isclose(lsf['fwhm'], 10 * 2 * np.sqrt(2 * np.log(2)), rtol=1e-3)
    # (a Hanning smoothing: on channels of which its width is a multiple)
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='convolution', lsfs=[
            dict(type='gauss', sigma=10), dict(type='hanning', width=10)]))
    ).json()
    assert (10 / lsf['step']) % 1 == 0, lsf['step']
    # (of an image of the project, relative to it)
    os.makedirs(os.path.join(root, 'study/data'), exist_ok=True)
    y, x = np.mgrid[-10:11, -10:11]
    fits.writeto(os.path.join(root, 'study/data/psf.fits'),
                 np.exp(-(x * x + y * y) / 8.0))
    image = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(
            type='image', file='data/psf.fits', step=[0.2, 0.2]))).json()
    assert np.isclose(np.array(image['values']).sum(), 1, atol=1e-4)
    for options, message in [
            (dict(type='gauss'), 'sigma'),
            (dict(type='gauss', sigma=-1), 'sigma'),
            (dict(type='image', file='../../x.fits'), 'out of the workspace')]:
        response = client.post('/api/v1/views/psf', json=dict(
            project='study', options=options))
        assert response.status_code == 400
        assert message in response.json()['detail'], response.json()
    assert client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='gauss', sigma=10),
        step=0.001)).status_code == 400


def test_the_parameters_of_a_galaxy_are_described(client):
    disk = dict(
        type='disk', rnodes=[0, 2, 4],
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            type='traits', vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=dict(type='lines'))
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(components=[disk]))).json()
    names = [p['name'] for p in info['parameters']]
    assert {'xpos', 'ypos', 'vsys', 'posa', 'incl', 'vpt_vt'} <= set(names)
    vsys = next(p for p in info['parameters'] if p['name'] == 'vsys')
    assert vsys['unit'] == 'km / s' and vsys['size'] == 1
    assert info['problems'] == []
    # (a galaxy of a band, seen in an image; one with problems)
    band = dict(disk, emission=dict(type='band'), kinematics=None)
    assert client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(components=[band]))).json()['parameters']
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(components=[dict(disk, tilted=True)])
    )).json()
    assert info['parameters'] == []
    assert info['problems'][0]['path'] == ['components', 0]


def test_the_schema_of_the_configurations_is_served(client):
    schema = client.get('/api/v1/schema').json()
    assert {'psf', 'lsf'} <= set(schema['$defs'])


def test_projects_are_renamed_and_deleted(client, root):
    response = client.patch('/api/v1/projects/study', json=dict(
        path='studies/ngc1313'))
    assert response.status_code == 200
    assert response.json()['pieces']['galaxies'] == ['disk']
    assert client.get('/api/v1/projects').json() == ['studies/ngc1313']
    client.post('/api/v1/projects', json=dict(path='other'))
    for path in ('other', 'other/inner', '.', '../out'):
        assert client.patch('/api/v1/projects/studies/ngc1313', json=dict(
            path=path)).status_code == 400, path
    assert client.patch('/api/v1/projects/nothing', json=dict(
        path='x')).status_code == 404
    # (a workspace that is a project stays)
    inner = Workspace(os.path.join(root, 'studies/ngc1313'))
    for change in (inner.delete_project, lambda p: inner.rename_project(p, 'x')):
        with pytest.raises(Invalid):
            change('.')
    assert client.delete('/api/v1/projects/other').status_code == 204
    assert not os.path.exists(os.path.join(root, 'other'))
    assert client.get('/api/v1/projects').json() == ['studies/ngc1313']


def test_folders_are_listed_and_files_of_data_described(client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CRVAL1=10, CDELT1=-1 / 3600, CUNIT1='deg',
        CTYPE2='DEC--TAN', CRVAL2=-30, CDELT2=1 / 3600, CUNIT2='deg',
        CTYPE3='VRAD', CRVAL3=0, CDELT3=10000, CUNIT3='m/s',
        BUNIT='Jy/beam', BMAJ=3 / 3600, BMIN=2 / 3600, BPA=30, RMS=0.002))
    fits.writeto(
        pathlib.Path(root, 'study/data/cube.fits'),
        np.zeros((4, 3, 2), np.float32), header)
    folder = client.get('/api/v1/folders/study/data').json()
    assert folder['folders'] == []
    assert [f['name'] for f in folder['files']] == ['cube.fits']
    assert client.get('/api/v1/folders/study/none').status_code == 404
    (array,) = client.get('/api/v1/describe/study/data/cube.fits').json()
    assert array['shape'] == [2, 3, 4]
    assert array['axes'] == ['RA---TAN', 'DEC--TAN', 'VRAD']
    assert [abs(step) for step in array['step']] == pytest.approx([1, 1, 10])
    assert array['beam'] == pytest.approx(dict(bmaj=3, bmin=2, bpa=30))
    assert array['rms'] == 0.002
    assert array['problems'] == []
    assert client.get(
        '/api/v1/describe/study/setups/disk.yaml').status_code == 400


def test_files_of_data_are_imported_and_uploaded_into_a_project(client, root):
    survey = pathlib.Path(root, 'survey')
    for name, value in (('cube.fits', 0), ('other/cube.fits', 1)):
        (survey / name).parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(survey / name, np.full((2, 2, 2), value, np.float32))

    def imported(path):
        return client.post('/api/v1/imports', json=dict(
            project='study', path=path)).json()

    assert imported('survey/cube.fits') == 'study/data/cube.fits'
    assert pathlib.Path(root, 'study/data/cube.fits').read_bytes() \
        == (survey / 'cube.fits').read_bytes()
    # (the same file again is the copy it has; another of its name gets a
    # number, and a file of the project stays where it is)
    assert imported('survey/cube.fits') == 'study/data/cube.fits'
    assert imported('survey/other/cube.fits') == 'study/data/cube-2.fits'
    assert imported('study/data/cube-2.fits') == 'study/data/cube-2.fits'

    def uploaded(name, content):
        return client.post(
            '/api/v1/uploads', params=dict(project='study', name=name),
            content=content)

    assert uploaded('sent.fits', b'SIMPLE').json() == 'study/data/sent.fits'
    assert uploaded('sent.fits', b'SIMPLE').json() == 'study/data/sent.fits'
    assert uploaded('sent.fits', b'OTHER').json() == 'study/data/sent-2.fits'
    assert uploaded('../sent.fits', b'').status_code == 400


def test_cubes_are_viewed_by_channel_spectrum_and_moments(client, root):
    # A line of flux 1 at 20 km/s in the spectra of a cube of 5 channels
    # of 10 km/s (from 0), with an axis of Stokes
    values = np.zeros((1, 5, 3, 4), np.float32)
    values[0, 2] = 1
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CRVAL1=10, CDELT1=-1 / 3600, CUNIT1='deg',
        CTYPE2='DEC--TAN', CRVAL2=-30, CDELT2=1 / 3600, CUNIT2='deg',
        CTYPE3='VRAD', CRVAL3=0, CDELT3=10000, CRPIX3=1, CUNIT3='m/s',
        CTYPE4='STOKES', CRVAL4=1, CDELT4=1, CRPIX4=1))
    fits.writeto(pathlib.Path(root, 'study/data/cube.fits'), values, header)
    path, query = 'study/data/cube.fits', dict(hdu=0)
    header, data = frames.decode(client.get(
        f'/api/v1/views/channel/{path}', params=query | dict(channel=2)
    ).content)
    (channel,) = frames.arrays(header, data)
    assert channel.shape == (3, 4) and np.all(channel == 1)
    assert (header['nchannels'], header['spectral'], header['unit']) == (
        5, 20, 'km/s')
    spectrum = client.get(
        f'/api/v1/views/spectrum/{path}', params=query | dict(x=1, y=2)
    ).json()
    assert spectrum['values'] == [0, 0, 1, 0, 0]
    assert spectrum['spectral'] == pytest.approx([0, 10, 20, 30, 40])
    assert spectrum['unit'] == 'km/s'
    # (the total spectrum: of the 12 pixels, or of a region of 2 x 2)
    assert client.get(
        f'/api/v1/views/spectrum/{path}', params=query
    ).json()['values'] == [0, 0, 12, 0, 0]
    assert client.get(
        f'/api/v1/views/spectrum/{path}',
        params=query | dict(x0=1, x1=2, y0=0, y1=1)
    ).json()['values'] == [0, 0, 4, 0, 0]
    assert client.get(
        f'/api/v1/views/spectrum/{path}',
        params=query | dict(x0=1, x1=9, y0=0, y1=1)
    ).status_code == 400
    header, data = frames.decode(client.get(
        f'/api/v1/views/moments/{path}', params=query | dict(snr=0)).content)
    assert [a['name'] for a in header['arrays']] == [
        'flux', 'velocity', 'dispersion', 'skewness', 'kurtosis']
    flux, velocity, dispersion, *_ = frames.arrays(header, data)
    np.testing.assert_allclose(flux, 10)
    np.testing.assert_allclose(velocity, 20)
    np.testing.assert_allclose(dispersion, 0, atol=1e-6)
    assert client.get(
        f'/api/v1/views/channel/{path}', params=query | dict(channel=5)
    ).status_code == 400


def test_a_map_is_viewed_as_a_cube_of_one_channel(client, root):
    # (a moment map, with an axis of length 1, as other codes write them)
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='VRAD', CDELT3=1000, CUNIT3='m/s'))
    fits.writeto(pathlib.Path(root, 'study/data/mom1.fits'), values, header)
    header, data = frames.decode(client.get(
        '/api/v1/views/channel/study/data/mom1.fits',
        params=dict(hdu=0, channel=0)).content)
    (plane,) = frames.arrays(header, data)
    np.testing.assert_array_equal(plane, values[0])
    assert (header['nchannels'], header['spectral']) == (1, None)


def test_the_moments_of_a_cube_are_of_the_mask_of_its_signal(client, root):
    # Noise of 1, and a line of 20 at 50 km/s in the spectra of a block of
    # 4 x 4 pixels, in 11 channels of 10 km/s (from 0)
    values = np.random.default_rng(1).normal(0, 1, (11, 12, 12))
    values[4:7, 4:8, 4:8] += 20
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='VRAD', CRVAL3=0, CDELT3=10000, CRPIX3=1,
        CUNIT3='m/s'))
    fits.writeto(
        pathlib.Path(root, 'study/data/line.fits'),
        values.astype(np.float32), header)
    header, data = frames.decode(client.get(
        '/api/v1/views/moments/study/data/line.fits',
        params=dict(hdu=0, snr=5)).content)
    flux, velocity, *_ = frames.arrays(header, data)
    # (the moments of the block, of its line alone; none elsewhere)
    block = np.zeros((12, 12), bool)
    block[4:8, 4:8] = True
    assert np.isfinite(flux[block]).all() and np.isnan(flux[~block]).all()
    np.testing.assert_allclose(flux[block], 600, rtol=0.05)
    np.testing.assert_allclose(velocity[block], 50, atol=1)


def test_a_job_prepares_a_cube_into_the_project(client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='VRAD', CDELT3=10000, CUNIT3='m/s'))
    values = np.full((5, 3, 4), -1, np.float32)
    values[2] = 1
    fits.HDUList([fits.PrimaryHDU(values, header), fits.ImageHDU(
        np.ones_like(values), name='STAT')]).writeto(
            pathlib.Path(root, 'study/data/cube.fits'))

    def prepare(clip_min):
        job = client.post('/api/v1/jobs', json=dict(
            task='prep_pixel_spectra', options=dict(
                file_d='study/data/cube.fits',
                file_e=dict(file='study/data/cube.fits', hdu='STAT'),
                error_kind='variance', file_m=None, roi_spat=None, roi_spec=None,
                clip_min=clip_min, clip_max=None, ccl_lcount=None,
                ccl_pcount=None, ccl_lratio=None, sclip_sigma=None,
                sclip_iters=None, minify=False, nanpad=None,
                dtype='float32', output_dir='study/data'))).json()
        for _ in range(600):
            job = client.get(f"/api/v1/jobs/{job['id']}").json()
            if job['state'] not in ('queued', 'running'):
                break
            time.sleep(0.1)
        assert job['state'] == 'complete', job['output']
        return job['files']
    assert prepare(0) == dict(
        file_d='study/data/prep_cube.fits',
        file_e='study/data/prep_cube_error.fits')
    prepared = fits.getdata(pathlib.Path(root, 'study/data/prep_cube.fits'))
    assert np.isnan(prepared[0]).all() and (prepared[2] == 1).all()
    # (prepared again, the same files are kept; others are numbered)
    assert prepare(0)['file_d'] == 'study/data/prep_cube.fits'
    assert prepare(None)['file_d'] == 'study/data/prep_cube-2.fits'
    assert sorted(os.listdir(pathlib.Path(root, 'study/data'))) == [
        'cube.fits', 'prep_cube-2.fits', 'prep_cube.fits',
        'prep_cube_error-2.fits', 'prep_cube_error.fits']


def test_a_job_prepares_a_flux_map_into_the_project(client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600))
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    fits.PrimaryHDU(values, header).writeto(
        pathlib.Path(root, 'study/data/image.fits'))
    job = client.post('/api/v1/jobs', json=dict(
        task='prep_pixel_brightness', options=dict(
            file_d='study/data/image.fits', file_e=None, file_m=None,
            roi_spat=[1, 3, 0, 3], clip_min=2, clip_max=None,
            ccl_lcount=None, ccl_pcount=None, ccl_lratio=None,
            sclip_sigma=None, sclip_iters=5, minify=False, nanpad=None,
            dtype='float32', output_dir='study/data'))).json()
    for _ in range(600):
        job = client.get(f"/api/v1/jobs/{job['id']}").json()
        if job['state'] not in ('queued', 'running'):
            break
        time.sleep(0.1)
    assert job['state'] == 'complete', job['output']
    assert job['files'] == dict(file_d='study/data/prep_image.fits')
    prepared = fits.getdata(pathlib.Path(root, 'study/data/prep_image.fits'))
    assert prepared.shape == (3, 2) and np.isnan(prepared[0, 0])
    assert prepared[2, 1] == 10


def test_a_job_prepares_moment_maps_with_one_mask(client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600))
    flux = np.ones((4, 5), np.float32)
    velocity = np.full((4, 5), 100, np.float32)
    velocity[0, 0] = np.nan
    velocity[3, 4] = 900
    folder = pathlib.Path(root, 'study/data')
    fits.writeto(folder / 'mom0.fits', flux, header)
    fits.HDUList([fits.PrimaryHDU(velocity, header), fits.ImageHDU(
        np.ones_like(velocity), name='ERR')]).writeto(folder / 'mom1.fits')
    job = client.post('/api/v1/jobs', json=dict(
        task='prep_pixel_moments', options=dict(
            file_d=['study/data/mom0.fits', 'study/data/mom1.fits'],
            file_e=[None, dict(file='study/data/mom1.fits', hdu='ERR')],
            file_m=None, roi_spat=None, clip_min=None,
            clip_max=[None, 500], ccl_lcount=None, ccl_pcount=None,
            ccl_lratio=None, sclip_sigma=[None, None], sclip_iters=5,
            minify=False, nanpad=None, dtype='float32',
            output_dir='study/data'))).json()
    for _ in range(600):
        job = client.get(f"/api/v1/jobs/{job['id']}").json()
        if job['state'] not in ('queued', 'running'):
            break
        time.sleep(0.1)
    assert job['state'] == 'complete', job['output']
    assert job['files'] == dict(
        file_d=['study/data/prep_mom0.fits', 'study/data/prep_mom1.fits'],
        file_e=[None, 'study/data/prep_mom1_error.fits'])
    # (a pixel left out of one map, as NaN or out of its range, is left
    # out of all)
    prepared = fits.getdata(folder / 'prep_mom0.fits')
    assert np.isnan(prepared[0, 0]) and np.isnan(prepared[3, 4])
    assert np.isfinite(prepared).sum() == 18


def test_the_files_of_moment_maps_are_lists(client):
    job = client.post('/api/v1/jobs', json=dict(
        task='prep_pixel_moments', options=dict(
            file_d='study/data/mom0.fits', output_dir='study/data'))).json()
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{job['id']}").json()
        if job['state'] not in ('queued', 'running'):
            break
        time.sleep(0.1)
    assert job['state'] == 'failed' and 'a list' in job['output']
