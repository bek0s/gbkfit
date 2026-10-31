"""
Tests for the API of the server: its token, and the round trips of a
project, a job and its run, and a live session.
"""

import json
import pathlib
import time

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient

from gbkfit_server.api import create_app, openapi
from gbkfit_server.services import frames
from gbkfit_server.services.executors import LocalExecutor
from gbkfit_server.services.server import Server


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
