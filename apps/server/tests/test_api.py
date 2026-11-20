"""
Tests for the API of the server: its token, and the round trips of a
project, a job and its run, and a live session.
"""

import copy
import json
import math
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
    assert 'type: thin_disk' in client.get(f'/api/v1/files/{path}').text
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


def test_a_project_gives_the_links_of_its_pieces(client):
    # (each piece or setup that names a piece, and whether that is missing:
    # the setup names its galaxy, observation and params; once a piece is
    # deleted, its name stays, missing)
    def link(folder, name, missing=False):
        return dict(
            source=dict(folder='setups', name='disk'),
            target=dict(folder=folder, name=name), missing=missing)
    assert client.get('/api/v1/projects/study').json()['links'] == [
        link('galaxies', 'disk'), link('observations', 'image'),
        link('params', 'disk')]
    client.delete('/api/v1/pieces/study/observations/image.yaml')
    assert link('observations', 'image', True) in client.get(
        '/api/v1/projects/study').json()['links']


def test_comparisons_are_pieces_of_the_studio_that_follow_their_names(
        client, root):
    # (the library does not know them: the workspace writes them, and they
    # follow the galaxies and the observations they name as those are
    # renamed)
    content = dict(galaxies=['disk'], observations=['image'], rows='galaxies')
    path = client.post('/api/v1/pieces', json=dict(
        project='study', folder='comparisons', name='all/first',
        content=content)).json()
    assert path == 'study/comparisons/all/first.yaml'
    assert client.get(f'/api/v1/pieces/{path}').json() == content
    assert client.get('/api/v1/projects/study').json()['pieces'][
        'comparisons'] == ['all/first']
    assert dict(folder='comparisons', name='all/first') in client.get(
        '/api/v1/references/study/galaxies/disk.yaml').json()
    client.patch('/api/v1/pieces/study/galaxies/disk.yaml',
                 json=dict(name='thin'))
    assert client.get(f'/api/v1/pieces/{path}').json()['galaxies'] == \
        ['thin']
    client.put(f'/api/v1/pieces/{path}', json=dict(content, rows='x'))
    assert client.get(f'/api/v1/pieces/{path}').json()['rows'] == 'x'
    path = client.patch(f'/api/v1/pieces/{path}', json=dict(name='second')) \
        .json()
    assert path == 'study/comparisons/second.yaml'
    assert not pathlib.Path(root, 'study/comparisons/all').exists()
    assert client.post('/api/v1/pieces', json=dict(
        project='study', folder='comparisons', name='../x')).status_code \
        == 400
    assert client.delete(f'/api/v1/pieces/{path}').status_code == 204
    assert client.get('/api/v1/projects/study').json()['pieces'][
        'comparisons'] == []


def test_a_setup_gives_each_parameter_the_file_of_its_layer(client, setup):
    # (as edited: a layer in place after the params piece of the galaxy
    # replaces its inclination; a parameter no layer gives has none)
    content = client.get(f'/api/v1/pieces/{setup}').json()
    content['params'] = ['disk', dict(properties=dict(incl=60))]
    info = client.post('/api/v1/setups/describe', json=dict(
        path=setup, setup=content)).json()
    assert info['problems'] == []
    by_name = {p['name']: p for p in info['parameters']}
    assert by_name['posa']['source'] == 'params/disk.yaml'
    assert by_name['posa']['properties'] == dict(posa=30)
    assert by_name['incl']['source'] == 'setups/disk.yaml'
    assert by_name['incl']['properties'] == dict(incl=60)
    assert by_name['incl']['item'] == '' and by_name['incl']['unit'] == 'deg'
    assert by_name['rpt_s']['source'] == 'params/disk.yaml'
    content['params'].append(dict(properties=dict(nothing=1)))
    problems = client.post('/api/v1/setups/describe', json=dict(
        path=setup, setup=content)).json()['problems']
    assert any('nothing' in p['message'] for p in problems)


def test_a_setup_is_observed_at_its_starts(client, setup):
    content = client.get(f'/api/v1/pieces/{setup}').json()
    observed = client.post('/api/v1/views/setup-observed', json=dict(
        path=setup, setup=content, observation=0)).json()
    assert observed['type'] == 'pixel_brightness'
    [model] = observed['arrays']
    assert model['kind'] == 'model' and model['shape'] == [16, 12, 1]
    assert client.post('/api/v1/views/setup-observed', json=dict(
        path=setup, setup=content, observation=1)).status_code == 400


def test_pieces_are_updated_and_their_comments_stay(client):
    path = client.post('/api/v1/pieces', json=dict(
        project='study', folder='psfs', name='beam')).json()
    # (the comment of the template, which says what it is, goes as it
    # changes; those of the user stay)
    content = dict(type='moffat', alpha=1.2, beta=3)
    assert client.put(f'/api/v1/pieces/{path}', json=content).status_code \
        == 204
    assert client.get(f'/api/v1/pieces/{path}').json() == content
    assert client.get(f'/api/v1/files/{path}').text == \
        'type: moffat\nalpha: 1.2\nbeta: 3\n'
    client.put(f'/api/v1/files/{path}', content='# mine\n' + client.get(
        f'/api/v1/files/{path}').text, headers={'content-type': 'text/plain'})
    client.put(f'/api/v1/pieces/{path}', json=dict(content, beta=4))
    assert client.get(f'/api/v1/files/{path}').text.startswith('# mine')
    assert client.put('/api/v1/pieces/study/psfs/no.yaml', json=content) \
        .status_code == 404


def test_files_are_written_and_can_be_emptied(client):
    # (e.g. notes cleared: no text writes an empty file)
    path = 'study/README.md'
    headers = {'content-type': 'text/plain'}
    assert client.put(
        f'/api/v1/files/{path}', content='# notes\n',
        headers=headers).status_code == 204
    assert client.get(f'/api/v1/files/{path}').text == '# notes\n'
    assert client.put(
        f'/api/v1/files/{path}', content='', headers=headers).status_code \
        == 204
    assert client.get(f'/api/v1/files/{path}').text == ''

def test_psfs_and_lsfs_are_drawn_with_their_widths(client, root):
    psf = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(type='gauss', sigma=1.0))).json()
    values = np.array(psf['values'])
    assert values.shape[0] == values.shape[1] and values.shape[0] > 100
    assert np.isclose(values.sum(), 1, atol=1e-4)
    assert np.isclose(psf['fwhm'], 2 * np.sqrt(2 * np.log(2)), rtol=1e-6)
    assert psf['axes'][2] is None
    assert np.allclose(psf['axes'][:2], psf['fwhm'], rtol=0.01)
    # (of an elliptical one, its widths along its axes, and its angle)
    beam = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(
            type='gauss', bmaj=6, bmin=3, bpa=30))).json()
    assert np.allclose(beam['axes'], [6, 3, 30], rtol=0.01)
    # (on the pixels of data)
    psf = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(type='gauss', sigma=1.0),
        step=[0.5, 0.5])).json()
    assert psf['step'] == [0.5, 0.5] and len(psf['values']) < 100
    point = client.post('/api/v1/views/psf', json=dict(
        project='study', options=dict(type='point'))).json()
    assert point['values'] == [[1.0]] and point['fwhm'] is None
    assert point['axes'] is None
    kms = '10 km/s'
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='gauss', sigma=kms))).json()
    assert np.isclose(lsf['fwhm'], 10 * 2 * np.sqrt(2 * np.log(2)), rtol=1e-3)
    # (a Hanning smoothing, of the channels it is drawn on: of the data)
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='convolution', lsfs=[
            dict(type='gauss', sigma=kms), dict(type='hanning')]),
        step=10)).json()
    assert lsf['step'] == 10
    # (a width in Angstrom spans km/s by the rest of a line, given)
    angstrom = dict(type='gauss', fwhm='2.5 Angstrom')
    response = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=angstrom))
    assert response.status_code == 400
    assert "its rest" in response.json()['detail']
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=angstrom, rest='6562.8 Angstrom')).json()
    assert np.isclose(lsf['fwhm'], 299792.458 * 2.5 / 6562.8, rtol=1e-3)
    response = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=angstrom, rest='6562.8'))
    assert response.status_code == 400
    assert "the rest of a spectral axis" in response.json()['detail']
    assert lsf['unit'] == 'km/s'
    # (or on channels of its own unit, of no rest)
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=angstrom, unit='Angstrom')).json()
    assert lsf['unit'] == 'Angstrom'
    assert np.isclose(lsf['fwhm'], 2.5, rtol=1e-3)
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='gauss', fwhm='2 pix'),
        unit='pix')).json()
    assert np.isclose(lsf['fwhm'], 2, rtol=1e-3)
    # (in pixels, of the channels of the data, drawn on finer ones)
    lsf = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='gauss', fwhm='2 pix'),
        channel=20)).json()
    assert lsf['step'] < 20 and np.isclose(lsf['fwhm'], 40, rtol=1e-3)
    hanning = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(type='hanning'), channel=20)).json()
    assert np.isclose(np.sum(hanning['values']), 1, atol=1e-4)
    for request, message in [
            (dict(unit='pix', channel=2), "of width 1"),
            (dict(unit='Angstrom', rest='6562.8 Angstrom'), "axis of velocities"),
            (dict(unit='Hz'), "spans no length")]:
        response = client.post('/api/v1/views/lsf', json=dict(
            project='study', options=angstrom) | request)
        assert response.status_code == 400
        assert message in response.json()['detail'], response.json()
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
    response = client.post('/api/v1/views/lsf', json=dict(
        project='study', options=dict(
            type='gauss', sigma='10 km/s'),
        step=0.001))
    assert response.status_code == 400
    assert "channels of 0.001 km/s" in response.json()['detail']


def test_each_type_of_component_is_given_with_its_parts(client):
    types = client.get('/api/v1/components').json()
    assert list(types) == [
        'thin_disk', 'thick_disk', 'cone', 'spheroid', 'jeans_spheroid',
        'orbit_spheroid', 'point', 'particles']
    # (in the order people read them, with the heights of its traits)
    parts = [
        (part['option'], part['heights'])
        for part in types['thick_disk']['parts']]
    assert parts[:3] == [
        ('rnodes', None), ('rptraits', 'rhtraits'), ('vptraits', 'vhtraits')]
    # (a type without traits takes none)
    assert [part['option'] for part in types['point']['parts']] \
        == ['emission']
    assert types['point']['traits'] == {}
    assert types['point']['directions'] == []


def test_the_traits_each_type_of_component_takes_are_given(client):
    takes = client.get('/api/v1/components').json()
    # (a cone takes only some density traits, and flows out or around)
    assert 'spiral' in takes['thin_disk']['traits']['rpt']
    assert 'spiral' not in takes['cone']['traits']['rpt']
    # (a thin disk has no heights but the vertical equilibrium that it does
    # not draw, warps out of its plane, or speeds away from it; a thick one
    # has them; the equilibrium of the dispersion is no trait, but the
    # option equilibrium of a thick disk)
    assert takes['thin_disk']['traits']['rht'] == ['equilibrium']
    assert 'equilibrium' in takes['thick_disk']['traits']['rht']
    assert 'equilibrium' not in takes['thick_disk']['traits']['dpt']
    for kind in ('vht', 'dht', 'zpt'):
        assert takes['thin_disk']['traits'][kind] == []
        assert takes['thick_disk']['traits'][kind]
    assert 'away' not in takes['thin_disk']['directions']
    assert 'away' in takes['thick_disk']['directions']
    assert set(takes['cone']['directions']) == {'tan', 'out'}
    assert takes['cone']['dispersion_directions'] == ['iso']
    assert 'mass' not in takes['spheroid']['traits']['vpt']


def test_the_parameters_of_a_galaxy_are_described(client):
    disk = dict(
        type='thin_disk', rnodes=[0, 2, 4],
        rptraits=dict(type='exponential'),
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'),
        emission=dict(type='lines'))
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(components=[disk]))).json()
    names = [p['name'] for p in info['parameters']]
    assert {'xpos', 'ypos', 'vsys', 'posa', 'incl', 'vpt_vt'} <= set(names)
    vsys = next(p for p in info['parameters'] if p['name'] == 'vsys')
    assert vsys['unit'] == 'km / s' and vsys['size'] == 1
    # Each with the option that makes it (see Galaxy.param_owners)
    owners = {p['name']: p['owner'] for p in info['parameters']}
    assert owners['vsys'] == ['components', 0]
    assert owners['vpt_vt'] == ['components', 0, 'vptraits', 0]
    assert info['problems'] == []
    # (a galaxy of a band, seen in an image; one with problems)
    band = dict(
        disk, emission=dict(type='band'), vptraits=None, dptraits=None)
    assert client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(components=[band]))).json()['parameters']
    # A thick disk, which needs a line of sight with a depth axis
    thick = dict(
        disk, type='thick_disk', rptraits=dict(type='exponential'),
        rhtraits=dict(type='sech2', zmax=6))
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(components=[thick]))).json()
    assert 'rht_s' in [p['name'] for p in info['parameters']]
    assert info['problems'] == []
    info = client.post('/api/v1/parameters', json=dict(
        project='study',
        galaxy=dict(components=[dict(disk, rnodes=[1, 2, 4])])
    )).json()
    assert info['parameters'] == []
    assert info['problems'][0]['path'] == ['components', 0]


def test_a_problem_in_a_part_of_a_galaxy_keeps_the_parameters_of_the_others(
        client):
    disk = dict(
        type='thin_disk', rnodes=[0, 2, 4],
        rptraits=dict(type='exponential'),
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'),
        emission=dict(type='lines'))
    galaxy = dict(
        components=[dict(disk, name='gas'), dict(disk, name='stars')],
        distance=dict(type='parameter'),
        mass=[dict(type='halo', name='halo', profile=dict(type='nfw')),
              dict(type='traced_disk', name='gas', component='gas')])
    whole = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=galaxy)).json()
    assert whole['problems'] == []
    # A part of the mass that follows no component: the galaxy is not
    # made, but its other parts are, each on its own, with the same names
    # and the paths of their options in the whole galaxy
    broken = dict(galaxy, mass=[galaxy['mass'][0], dict(
        galaxy['mass'][1], component='nowhere')])
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=broken)).json()
    assert [p['path'][:2] for p in info['problems']] == [['mass', 1]]
    assert info['parameters'] == [
        p for p in whole['parameters'] if p['owner'] != ['mass', 1]]
    # A component with a problem: the other one, the distance and the halo
    bad = dict(disk, name='stars', rnodes=[1, 2, 4])
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=dict(galaxy, components=[
            galaxy['components'][0], bad]))).json()
    owners = {p['name']: p['owner'] for p in info['parameters']}
    assert owners['gas_vpt_vt'] == ['components', 0, 'vptraits', 0]
    assert owners['mass_halo_c'] == ['mass', 0]
    assert 'distance' in owners
    assert not any(name.startswith('stars_') for name in owners)


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
    # (the header as it is in the file, of each HDU)
    cards = client.get('/api/v1/header/study/data/cube.fits').json()['cards']
    assert any(card.startswith("BUNIT   = 'Jy/beam") for card in cards)
    assert client.get(
        '/api/v1/header/study/data/cube.fits?hdu=3').status_code == 404
    # (of velocities of no convention, the size of its pixels on the sky
    # alone, until one is given)
    header['CTYPE3'] = 'VELO-HEL'
    fits.writeto(
        pathlib.Path(root, 'study/data/velo.fits'),
        np.zeros((4, 3, 2), np.float32), header)
    (array,) = client.get('/api/v1/describe/study/data/velo.fits').json()
    assert array['ambiguous_velocity']
    assert [abs(step) for step in array['step']] == pytest.approx([1, 1])
    (array,) = client.get(
        '/api/v1/describe/study/data/velo.fits?velocity=radio').json()
    assert [abs(step) for step in array['step']] == pytest.approx([1, 1, 10])


def test_a_cube_of_wavelengths_is_described_in_velocities_of_a_rest(
        client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-0.2 / 3600, CUNIT1='deg',
        CTYPE2='DEC--TAN', CDELT2=0.2 / 3600, CUNIT2='deg',
        CTYPE3='AWAV', CRVAL3=6600, CDELT3=1.25, CUNIT3='Angstrom'))
    fits.writeto(
        pathlib.Path(root, 'study/data/ifu.fits'),
        np.zeros((4, 3, 2), np.float32), header)
    # (without a rest, the size of its pixels on the sky alone, and the
    # problem of its axis)
    (array,) = client.get('/api/v1/describe/study/data/ifu.fits').json()
    assert array['step'] == pytest.approx([0.2, 0.2])
    assert array['problems'] == [dict(
        severity='error', path=['spectral', 'rest'],
        message='AWAV axis: rest missing')]
    assert array['rest'] is None
    (array,) = client.get(
        '/api/v1/describe/study/data/ifu.fits',
        params=dict(rest='6562.8 Angstrom')).json()
    assert array['problems'] == [] and array['rest'] == '6562.8 Angstrom'
    assert [abs(step) for step in array['step']] == pytest.approx(
        [0.2, 0.2, 299792.458 * 1.25 / 6562.8], rel=1e-3)
    # (an axis of a type FITS does not define, as CALIFA writes: what it
    # is is said, never taken from its name)
    header['CTYPE3'] = 'WAVELENGTH'
    fits.writeto(
        pathlib.Path(root, 'study/data/califa.fits'),
        np.zeros((4, 3, 2), np.float32), header)
    rest = dict(rest='6562.8 Angstrom')
    (array,) = client.get(
        '/api/v1/describe/study/data/califa.fits', params=rest).json()
    assert array['problems'] == [dict(
        severity='error', path=['spectral', 'type'],
        message="CTYPE3 'WAVELENGTH' not FITS")]
    (array,) = client.get(
        '/api/v1/describe/study/data/califa.fits',
        params=rest | dict(spectral_type='AWAV')).json()
    assert array['problems'] == []
    assert abs(array['step'][2]) == pytest.approx(
        299792.458 * 1.25 / 6562.8, rel=1e-3)
    assert client.get(
        '/api/v1/describe/study/data/califa.fits',
        params=dict(spectral_type='WAVELENGTH')).status_code == 422
    # (the rest of a header, as a radio cube of velocities states it)
    radio = fits.Header(dict(
        CTYPE1='RA---SIN', CDELT1=-2 / 3600, CUNIT1='deg',
        CTYPE2='DEC--SIN', CDELT2=2 / 3600, CUNIT2='deg',
        CTYPE3='VRAD', CRVAL3=1000, CDELT3=4000, CUNIT3='m/s',
        RESTFRQ=1.420405752e9, SPECSYS='BARYCENT'))
    fits.writeto(
        pathlib.Path(root, 'study/data/radio.fits'),
        np.zeros((4, 3, 2), np.float32), radio)
    (array,) = client.get('/api/v1/describe/study/data/radio.fits').json()
    assert array['rest'] == '1.420406 GHz'


def wait(client, job):
    """Return a job once it ends."""
    for _ in range(600):
        job = client.get(f"/api/v1/jobs/{job['id']}").json()
        if job['state'] not in ('queued', 'running'):
            return job
        time.sleep(0.1)
    raise AssertionError(f"the job {job['id']} does not end")


def test_a_preview_of_prep_is_kept_and_its_preparation_recorded(
        client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600))
    folder = pathlib.Path(root, 'study/data')
    fits.writeto(
        folder / 'image.fits',
        np.arange(12, dtype=np.float32).reshape(3, 4), header)
    options = dict(
        file_d='study/data/image.fits', file_e=None, file_m=None,
        roi_spat=[1, 3, 0, 3], clip_min=None, clip_max=None,
        ccl_lcount=None, ccl_pcount=None, ccl_lratio=None,
        sclip_sigma=None, sclip_iters=5, minify=False, nanpad=None,
        dtype='float32', output_dir='study/data')
    job = wait(client, client.post('/api/v1/jobs', json=dict(
        task='prep_pixel_brightness', options=options, preview=True)).json())
    assert job['state'] == 'complete' and job['preview']
    # (its files stay apart, out of the listing of the folder)
    preview = job['files']['file_d']
    assert preview.startswith('study/data/.prep-')
    assert fits.getdata(pathlib.Path(root, preview)).shape == (3, 2)
    assert [f['name'] for f in client.get(
        '/api/v1/folders/study/data').json()['files']] == ['image.fits']
    kept = client.post(f"/api/v1/jobs/{job['id']}/keep").json()
    assert not kept['preview']
    assert kept['files'] == dict(file_d='study/data/prep_image.fits')
    assert not pathlib.Path(root, preview).exists()
    # (kept once)
    assert client.post(
        f"/api/v1/jobs/{job['id']}/keep").status_code == 400
    record = client.get(
        '/api/v1/preparations/study/data/prep_image.fits').json()
    assert record['task'] == 'prep_pixel_brightness'
    assert record['options']['file_d'] == 'study/data/image.fits'
    assert record['options']['roi_spat'] == [1, 3, 0, 3]
    assert 'output_dir' not in record['options']
    assert record['files'] == kept['files']
    assert client.get(
        '/api/v1/preparations/study/data/image.fits').json() is None
    # (the record follows the project as it moves)
    client.patch('/api/v1/projects/study', json=dict(path='moved'))
    record = client.get(
        '/api/v1/preparations/moved/data/prep_image.fits').json()
    assert record['options']['file_d'] == 'moved/data/image.fits'


def test_previews_not_kept_are_removed_as_newer_ones_end(client, root):
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600))
    fits.writeto(
        pathlib.Path(root, 'study/data/image.fits'),
        np.ones((3, 4), np.float32), header)
    jobs = [wait(client, client.post('/api/v1/jobs', json=dict(
        task='prep_pixel_brightness', preview=True, options=dict(
            file_d='study/data/image.fits', file_e=None, file_m=None,
            roi_spat=None, clip_min=clip, clip_max=None, ccl_lcount=None,
            ccl_pcount=None, ccl_lratio=None, sclip_sigma=None,
            sclip_iters=5, minify=False, nanpad=None, dtype='float32',
            output_dir='study/data'))).json())
        for clip in range(4)]
    first = client.get(f"/api/v1/jobs/{jobs[0]['id']}").json()
    assert first['files'] is None
    assert not pathlib.Path(root, jobs[0]['files']['file_d']).exists()
    assert all(pathlib.Path(root, job['files']['file_d']).exists()
               for job in jobs[1:])
    assert client.post(
        f"/api/v1/jobs/{jobs[0]['id']}/keep").status_code == 400
    assert client.post('/api/v1/jobs', json=dict(
        task='fit', preview=True,
        options=dict(config='study/setups/disk.yaml'))).status_code == 400


def test_files_of_data_are_deleted_unless_the_project_names_them(
        client, root):
    folder = pathlib.Path(root, 'study/data')
    fits.writeto(folder / 'used.fits', np.ones((3, 4), np.float32))
    fits.writeto(folder / 'free.fits', np.ones((3, 4), np.float32))
    client.post('/api/v1/pieces', json=dict(
        project='study', folder='datasets', name='image', content=dict(
            type='pixel_brightness', data='data/used.fits')))
    response = client.delete('/api/v1/data/study/data/used.fits')
    assert response.status_code == 400
    assert 'datasets/image.yaml' in response.json()['detail']
    assert client.delete(
        '/api/v1/data/study/data/free.fits').status_code == 204
    assert not (folder / 'free.fits').exists()
    assert client.delete(
        '/api/v1/data/study/data/free.fits').status_code == 404
    # (only files of data of the folder data of a project)
    assert client.delete(
        '/api/v1/data/study/datasets/image.yaml').status_code == 400


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
    assert header['extent'] == [0, 1]
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
    # (of a Gauss-Hermite series, h3 and h4 alone)
    header, data = frames.decode(client.get(
        f'/api/v1/views/moments/{path}',
        params=query | dict(snr=0, method='gauss_hermite')).content)
    assert [a['name'] for a in header['arrays']] == ['h3', 'h4']
    assert all(a.shape == flux.shape for a in frames.arrays(header, data))
    assert client.get(
        f'/api/v1/views/channel/{path}', params=query | dict(channel=5)
    ).status_code == 400


def test_every_view_of_a_cube_reads_its_spectral_axis_as_given(
        client, root):
    # A line in the second channel of a cube of frequencies that rise along
    # the file: seen in GHz as they are; of a rest, in velocities, its
    # channels reversed to increase (as a dataset of it has them), in every
    # view alike
    rest = 1420.405752e6
    values = np.zeros((5, 3, 4), np.float32)
    values[1] = 1
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='FREQ', CRVAL3=rest, CRPIX3=3, CDELT3=1e5,
        CUNIT3='Hz'))
    fits.writeto(pathlib.Path(root, 'study/data/hi.fits'), values, header)
    views = '/api/v1/views/{}/study/data/hi.fits'
    spectrum = client.get(
        views.format('spectrum'), params=dict(hdu=0)).json()
    assert spectrum['unit'] == 'GHz'
    assert spectrum['values'] == [0, 12, 0, 0, 0]
    reading = dict(hdu=0, rest='1420.405752 MHz')
    spectrum = client.get(views.format('spectrum'), params=reading).json()
    width = 299792.458 * 1e5 / rest
    assert spectrum['unit'] == 'km/s'
    assert spectrum['values'] == [0, 0, 0, 12, 0]
    assert spectrum['spectral'] == pytest.approx(width * np.arange(-2, 3))
    header, data = frames.decode(client.get(
        views.format('channel'), params=reading | dict(channel=3)).content)
    (channel,) = frames.arrays(header, data)
    assert np.all(channel == 1)
    assert header['spectral'] == pytest.approx(width)
    header, data = frames.decode(client.get(
        views.format('moments'), params=reading | dict(snr=0)).content)
    np.testing.assert_allclose(frames.arrays(header, data)[1], width)
    # Velocities in km/s in a header without a unit: in m/s (SI) as FITS
    # reads them, or in the unit given
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3='VRAD', CRVAL3=0, CRPIX3=1, CDELT3=10))
    fits.writeto(pathlib.Path(root, 'study/data/kms.fits'), values, header)
    path = '/api/v1/views/spectrum/study/data/kms.fits'
    assert client.get(path, params=dict(hdu=0)).json()['spectral'] == \
        pytest.approx([0, 0.01, 0.02, 0.03, 0.04])
    assert client.get(
        path, params=dict(hdu=0, spectral_unit='km/s')
    ).json()['spectral'] == pytest.approx([0, 10, 20, 30, 40])
    response = client.get(path, params=dict(hdu=0, spectral_unit='Jy'))
    assert response.status_code == 400 and 'Jy' in response.text
    response = client.get(
        path, params=dict(hdu=0, spectral_unit='Angstrom'))
    assert response.status_code == 400
    assert response.json()['detail'] == "CUNIT3 'Angstrom' not of VRAD"
    # A cube without CTYPE3: described as one whose type must be given; seen
    # in channels alone until it is, then as of the type and unit given
    del header['CTYPE3']
    fits.writeto(pathlib.Path(root, 'study/data/untyped.fits'), values, header)
    (array,) = client.get(
        '/api/v1/describe/study/data/untyped.fits').json()
    assert array['problems'][0] == dict(
        severity='error', path=['spectral', 'type'], message='CTYPE3 missing')
    path = '/api/v1/views/spectrum/study/data/untyped.fits'
    assert client.get(path, params=dict(hdu=0)).json()['spectral'] is None
    assert client.get(path, params=dict(
        hdu=0, spectral_type='VRAD', spectral_unit='km/s'
    )).json()['spectral'] == pytest.approx([0, 10, 20, 30, 40])


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
    assert header['extent'] == [0, 11]


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


@pytest.mark.parametrize('ctype, cdelt, rest', [
    ('AWAV', 1.0, '6562.8 Angstrom'), ('AWAV', -1.0, '6562.8 Angstrom'),
    ('FREQ', 1e5, '1420.405752 MHz'), ('FREQ', -1e5, '1420.405752 MHz')])
def test_the_moments_of_a_cube_of_wavelengths_are_of_a_rest(
        client, root, ctype, cdelt, rest):
    # A line in one channel of a cube of wavelengths or frequencies about
    # its rest, increasing or decreasing: its velocity, of the rest (as a
    # dataset of it reads it), wherever its channels go
    unit = 'Angstrom' if ctype == 'AWAV' else 'Hz'
    centre = 6562.8 if ctype == 'AWAV' else 1420.405752e6
    values = np.zeros((9, 3, 3), np.float32)
    values[6] = 1
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CDELT1=-1 / 3600, CTYPE2='DEC--TAN',
        CDELT2=1 / 3600, CTYPE3=ctype, CRVAL3=centre, CDELT3=cdelt,
        CRPIX3=5, CUNIT3=unit))
    fits.writeto(pathlib.Path(root, 'study/data/waves.fits'), values, header)
    path = '/api/v1/views/moments/study/data/waves.fits'
    response = client.get(path, params=dict(hdu=0, snr=0))
    assert response.status_code == 400
    assert "rest of a line" in response.json()['detail']
    header, data = frames.decode(client.get(
        path, params=dict(hdu=0, snr=0, rest=rest)).content)
    velocity = frames.arrays(header, data)[1]
    # (channel 6 is 2 channels from the rest: of a wavelength 2 Angstrom
    # above, of a frequency 2e5 Hz above, as the step goes)
    light = 299792.458
    expected = light * 2 * cdelt / centre if ctype == 'AWAV' \
        else -light * 2 * cdelt / centre
    np.testing.assert_allclose(velocity, expected, rtol=1e-3)


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
        '.prepared.json', 'cube.fits', 'prep_cube-2.fits', 'prep_cube.fits',
        'prep_cube_error-2.fits', 'prep_cube_error.fits']
    # (how each was prepared is recorded)
    record = client.get(
        '/api/v1/preparations/study/data/prep_cube-2.fits').json()
    assert record['options']['clip_min'] is None
    assert record['options']['file_e'] == dict(
        file='study/data/cube.fits', hdu='STAT')


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



def test_a_galaxy_of_mass_alone_is_described_and_drawn(client):
    # (its rotation curve is fitted; it emits no light)
    galaxy = dict(distance=dict(type='parameter'), mass=[dict(
        type='disk', name='disk', surface=dict(type='exponential'),
        vertical=dict(type='thin'))])
    info = client.post('/api/v1/parameters', json=dict(
        project='study', galaxy=galaxy)).json()
    assert {p['name'] for p in info['parameters']} == {
        'distance', 'mass_disk_log_m', 'mass_disk_rd'}
    assert info['problems'] == [] and info['whole']
    # Its curves, those of its mass, to 30 kpc; no maps
    header, maps = render(client, galaxy, properties=dict(
        distance=10, mass_disk_log_m=10, mass_disk_rd=3), size=64)
    assert maps == {}
    curves = header['curves']
    assert {'vcirc', 'vcirc_disk', 'mass'} <= set(curves)
    kpc = 10e3 * np.pi / 648000
    assert np.isclose(curves['radius'][-1] * kpc, 30)
    assert max(curves['vcirc']) > 50
    # (its light, in 3d, is nothing to see)
    response = client.post('/api/v1/views/galaxy-volume', json=dict(
        project='study', galaxy=galaxy, size=32))
    assert response.status_code == 400
    assert 'no component that emits light' in response.text

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


def render(client, galaxy, **request):
    """The header and the maps (by name) of a render of a galaxy."""
    response = client.post('/api/v1/views/galaxy', json=dict(
        project='study', galaxy=galaxy) | request)
    assert response.status_code == 200, response.text
    header, data = frames.decode(response.content)
    maps = dict(zip(
        [info['name'] for info in header['arrays']],
        frames.arrays(header, data)))
    return header, maps


# A rotating gas disk and a stellar disk in a band
RENDERED = dict(components=[
    dict(type='thin_disk', name='gas', rnodes=dict(max=10, sep=2),
         rptraits=dict(type='exponential'),
         vptraits=dict(type='arctan', direction='tan'),
         dptraits=dict(type='uniform'),
         emission=dict(type='lines')),
    dict(type='thin_disk', name='stars', rnodes=[0, 2, 4, 6, 8],
         rptraits=dict(type='exponential'),
         emission=dict(type='band'))])


def test_a_render_takes_keys_that_select_elements(client):
    # A warp whose inner rings are tied to another disk (keys with slices
    # and ties, as in parameter files), the elements of a vector each
    # given once
    warped = copy.deepcopy(RENDERED)
    warped['components'][0]['tilted'] = True
    _, sliced = render(client, warped, properties={
        'gas_vpt_vt': 200, 'stars_posa': 30, 'gas_posa[0:3]': 'stars_posa',
        'gas_posa[3:]': 40}, size=32)
    _, plain = render(client, warped, properties=dict(
        gas_vpt_vt=200, stars_posa=30, gas_posa=[30, 30, 30, 40, 40, 40]),
        size=32)
    for name in plain:
        np.testing.assert_array_equal(sliced[name], plain[name])


def test_a_galaxy_is_rendered_as_it_is_built(client):
    header, maps = render(
        client, RENDERED, properties=dict(gas_vpt_vt=200, gas_posa=0),
        size=64)
    assert {'light', 'velocity', 'dispersion', 'gas/light', 'gas/velocity',
            'stars/light'} <= set(maps)
    assert 'stars/velocity' not in maps
    assert maps['light'].shape == (64, 64)
    # The field holds the largest radius (10 arcsec, with a margin)
    assert header['size'] * header['step'] > 20
    # Its light is that of both disks, and it rotates about its centre
    np.testing.assert_allclose(
        maps['light'], maps['gas/light'] + maps['stars/light'], rtol=1e-5)
    velocity = maps['velocity']
    assert np.nanmax(velocity) > 50
    np.testing.assert_allclose(
        np.nanmax(velocity), -np.nanmin(velocity), rtol=0.05)
    # Its curves: the rotation and the dispersion of the gas disk
    curves = header['curves']
    assert len(curves['rotation_gas']) == len(curves['radius'])
    assert curves['rotation_gas'][-1] > 100
    assert header['problems'] == []


def test_a_galaxy_of_a_named_line_is_rendered(client):
    # (its cube at the rest of the line, given as text)
    gas = dict(RENDERED['components'][0], emission=dict(
        type='lines', lines='hi21'))
    _, maps = render(
        client, dict(components=[gas]), properties=dict(gas_vpt_vt=200),
        size=32)
    assert np.nanmax(maps['gas/velocity']) > 0


def test_the_curves_of_a_disk_are_along_each_direction(client):
    # (its speeds along each direction it has traits along, and its
    # density)
    gas = dict(RENDERED['components'][0], vptraits=[
        dict(type='arctan', direction='tan'),
        dict(type='uniform', direction='rad')])
    header, _ = render(
        client, dict(components=[gas]),
        properties=dict(gas_vpt_vt=200, gas_vpt2_vr=-8), size=32)
    curves = header['curves']
    assert {'speed_gas:tan', 'speed_gas:rad', 'density_gas',
            'rotation_gas'} <= set(curves)
    assert 'speed_gas:ver' not in curves
    assert curves['speed_gas:rad'] \
        == pytest.approx([-8] * len(curves['radius']))
    assert curves['speed_gas:tan'] == pytest.approx(curves['rotation_gas'])
    # (and its position angle and inclination)
    assert curves['orientation_gas:incl'] == pytest.approx(
        [curves['orientation_gas:incl'][0]] * len(curves['radius']))
    assert 'orientation_gas:posa' in curves


def test_a_galaxy_is_rendered_in_3d(client):
    # (face-on: its light is thickest at its plane, along z)
    thick = dict(
        RENDERED['components'][0], type='thick_disk',
        rhtraits=dict(type='sech2', zmax=4))
    response = client.post('/api/v1/views/galaxy-volume', json=dict(
        project='study', galaxy=dict(components=[thick]),
        properties=dict(gas_vpt_vt=200), size=24))
    assert response.status_code == 200, response.text
    header, data = frames.decode(response.content)
    light, = frames.arrays(header, data)
    assert light.ndim == 3 and light.shape[1:] == (24, 24)
    profile = light.sum(axis=(1, 2))
    assert abs(int(profile.argmax()) - header['zero']) <= 1
    assert header['depth_step'] > 0 and header['step'] > 0


def test_the_extra_outputs_of_a_galaxy_are_listed_and_rendered(client):
    header, _ = render(
        client, RENDERED, properties=dict(gas_vpt_vt=200), size=32)
    velocity = next(
        x for x in header['extras']
        if x['name'] == 'vdata' and x['ids'].get('component') == 'gas')
    assert velocity['title'] and velocity['ndim'] == 2
    response = client.post('/api/v1/views/galaxy-extra', json=dict(
        project='study', galaxy=RENDERED, properties=dict(gas_vpt_vt=200),
        size=32, name='vdata', ids=velocity['ids']))
    assert response.status_code == 200, response.text
    extra_header, data = frames.decode(response.content)
    values, = frames.arrays(extra_header, data)
    assert values.shape == (32, 32)
    assert extra_header['title'] == velocity['title']
    response = client.post('/api/v1/views/galaxy-extra', json=dict(
        project='study', galaxy=RENDERED, size=32, name='none'))
    assert response.status_code == 400


def test_a_galaxy_is_rendered_face_on_and_edge_on(client):
    _, face_on = render(
        client, RENDERED, properties=dict(gas_vpt_vt=200), view='face-on',
        size=64)
    # No motion along the line of sight, and its motion in the plane of
    # the sky (vx, vy)
    assert np.nanmax(np.abs(face_on['velocity'])) < 1e-3
    assert np.nanmax(np.hypot(face_on['vx'], face_on['vy'])) > 100
    _, edge_on = render(client, RENDERED, view='edge-on', size=64)
    # A thin disk edge-on has no light
    assert np.all(edge_on['gas/light'] == 0)


def test_a_warped_disk_keeps_its_warp_face_on_and_edge_on(client):
    # The galaxy turns as a whole, by its inner ring: its outer rings,
    # tilted from it, stay tilted (light edge-on, motion along the line of
    # sight face-on), and its volume is that of the sky turned face-on
    warped = copy.deepcopy(RENDERED)
    warped['components'][0]['tilted'] = True
    properties = dict(
        gas_vpt_vt=200, gas_posa=[30, 30, 30, 40, 50, 60],
        gas_incl=[60, 60, 60, 70, 80, 85])
    _, edge_on = render(
        client, warped, properties=properties, view='edge-on', size=64)
    assert edge_on['gas/light'].max() > 0
    _, face_on = render(
        client, warped, properties=properties, view='face-on', size=64)
    assert np.nanmax(np.abs(face_on['velocity'])) > 10


def test_a_galaxy_is_observed_by_an_observation_of_its_project(client):
    galaxy = client.get('/api/v1/pieces/study/galaxies/disk.yaml').json()
    info = client.post('/api/v1/views/observed', json=dict(
        project='study', galaxy=galaxy, properties=dict(rpt_s=2),
        observation='image')).json()
    assert info['type'] == 'pixel_brightness'
    [array] = info['arrays']
    assert (array['kind'], array['item'], array['shape']) \
        == ('model', 'brightness', [16, 12, 1])
    # (its pixels on the sky: of 1 arcsec, measured from the middle)
    assert array['step'] == [1, 1] and array['rpix'] == [7.5, 5.5]
    # (its map, as a channel of a cube, has the light of the disk)
    header, data = frames.decode(client.get(
        f"/api/v1/views/channel/{array['path']}",
        params=dict(hdu=0, channel=0)).content)
    assert frames.arrays(header, data)[0].max() > 0
    # (an observation it cannot be, or one not there, is refused)
    assert client.post('/api/v1/views/observed', json=dict(
        project='study', galaxy=galaxy, observation='none')
    ).status_code in (400, 404)


def test_the_cube_of_a_galaxy_is_rendered_and_viewed(client):
    info = client.post('/api/v1/views/galaxy-cube', json=dict(
        project='study', galaxy=RENDERED,
        properties=dict(gas_vpt_vt=200, gas_posa=0), size=48)).json()
    nx, ny, nchannels = info['shape']
    assert (nx, ny) == (48, 48) and 2 < nchannels <= 128
    assert info['unit'] == 'km/s'
    # Its channels hold the light of the gas: none at their ends, and its
    # total spectrum of two horns about 0
    spectrum = client.get(
        f"/api/v1/views/spectrum/{info['path']}", params=dict(hdu=0)).json()
    values = np.array(spectrum['values'])
    velocities = np.array(spectrum['spectral'])
    assert values[0] < 1e-3 * values.max() and values[-1] < 1e-3 * values.max()
    assert velocities[0] < -200 < 200 < velocities[-1]
    # (each horn the brightest channel on its side of 0)
    below, above = velocities < 0, velocities > 0
    np.testing.assert_allclose(
        velocities[below][np.argmax(values[below])],
        -velocities[above][np.argmax(values[above])],
        atol=2 * (velocities[1] - velocities[0]))
    # Its channels and moments, as those of a cube of a file
    response = client.get(
        f"/api/v1/views/channel/{info['path']}",
        params=dict(hdu=0, channel=nchannels // 2))
    assert response.status_code == 200
    response = client.get(
        f"/api/v1/views/moments/{info['path']}", params=dict(hdu=0, snr=0))
    assert response.status_code == 200
    # Its position-velocity diagram along its major axis: the light of the
    # receding side (towards its position angle) at positive velocities,
    # that of the other at negative ones
    response = client.get(
        f"/api/v1/views/pv/{info['path']}", params=dict(hdu=0, pa=0))
    assert response.status_code == 200, response.text
    header, data = frames.decode(response.content)
    pv, = frames.arrays(header, data)
    positions = np.array(header['positions'])
    spectral = np.array(header['spectral'])
    assert pv.shape == (nchannels, len(positions))
    assert header['unit'] == 'km/s' and header['step'] == info['step'] > 0
    far = np.argmin(np.abs(positions - 12)), np.argmin(np.abs(positions + 12))
    assert spectral[np.nanargmax(pv[:, far[0]])] > 50
    assert spectral[np.nanargmax(pv[:, far[1]])] < -50
    # At a redshift, its light is in its channels all the same: at the
    # velocities where its lines land (here of a frequency, radio ones)
    for line in (None, 'hi21'):
        gas = dict(RENDERED['components'][0])
        if line:
            gas['emission'] = dict(type='lines', lines=line)
        info = client.post('/api/v1/views/galaxy-cube', json=dict(
            project='study', galaxy=dict(components=[gas], redshift=0.002),
            properties=dict(gas_vpt_vt=200, gas_posa=0), size=32)).json()
        values = np.array(client.get(
            f"/api/v1/views/spectrum/{info['path']}", params=dict(hdu=0)
        ).json()['values'])
        assert max(values[0], values[-1]) < 1e-3 * values.max()
        assert (info['convention'], info['shift']) == (
            ('radio', 1.002) if line else ('relativistic', 1))
    # A galaxy without lines has no cube
    stars = dict(components=[RENDERED['components'][1]])
    response = client.post('/api/v1/views/galaxy-cube', json=dict(
        project='study', galaxy=stars))
    assert response.status_code == 400 and 'emits lines' in response.text
    # A cube no longer kept is asked for again
    response = client.get(
        '/api/v1/views/spectrum/@render/999', params=dict(hdu=0))
    assert response.status_code == 400 and 'make it again' in response.text


@pytest.mark.parametrize('galaxy, request_, message', [
    (dict(components=[]), {}, "at least one component"),
    (RENDERED, dict(properties=dict(nonsense=1)), "unknown parameters"),
    (RENDERED, dict(view='sideways'), None)])
def test_invalid_renders_are_refused(client, galaxy, request_, message):
    response = client.post('/api/v1/views/galaxy', json=dict(
        project='study', galaxy=galaxy) | request_)
    assert response.status_code in (400, 422)
    if message is not None:
        assert message in response.text


def test_the_lines_of_the_catalogue_are_served(client):
    lines = {line['name']: line for line in client.get(
        '/api/v1/lines').json()}
    assert lines['halpha']['rest'] == '6564.61 Angstrom'
    assert lines['hi21']['rest'] == '1.420406 GHz'
    assert lines['paalpha']['rest'] == '1.87561 um'
    # (in vacuum and in air: Halpha is at 6562.80 Angstrom in air)
    halpha = lines['halpha']
    assert halpha['wavelength'] == pytest.approx(6564.61e-10)
    assert halpha['air'] == pytest.approx(6562.797e-10, abs=1e-14)
    assert halpha['frequency'] is None
    hi21 = lines['hi21']
    assert hi21['frequency'] == pytest.approx(1420405751.768)
    assert hi21['wavelength'] is None and hi21['air'] is None


def test_the_preferences_of_the_user_are_set_and_read(client, user_home):
    info = client.get('/api/v1/preferences').json()
    assert info == dict(file=str(user_home / 'preferences.json'), values={})
    response = client.patch(
        '/api/v1/preferences', json={'theme': 'clay', 'colour-map': dict(
            map='viridis', inverted=False)})
    assert response.status_code == 200
    client.patch('/api/v1/preferences', json=dict(theme='amber'))
    assert client.get('/api/v1/preferences').json()['values'] == {
        'theme': 'amber', 'colour-map': dict(map='viridis', inverted=False)}
    assert client.patch(
        '/api/v1/preferences', json=['clay']).status_code == 422


# A lens of an SIE and a shear, its strengths quoted for a source at 1.5
LENS = dict(name='lens', redshift=0.5, source_redshift=1.5, pieces=[
    dict(type='sie'), dict(type='shear')])
LENS_PROPERTIES = dict(
    sie_theta_e=1.2, sie_q=1, sie_posa=0, sie_xpos=0, sie_ypos=0,
    shear_gamma=0, shear_posa=0)


def test_the_parameters_of_a_lens_are_described(client):
    # (by the names of the lens: of each piece, by its name; each with its
    # piece; a lens with a problem has none)
    info = client.post('/api/v1/lenses/describe', json=dict(
        project='study', lens=LENS)).json()
    owners = {p['name']: p['owner'] for p in info['parameters']}
    assert owners['sie_theta_e'] == ['pieces', 0]
    assert owners['shear_gamma'] == ['pieces', 1]
    assert info['problems'] == []
    info = client.post('/api/v1/lenses/describe', json=dict(
        project='study', lens=LENS | dict(source_redshift=None))).json()
    assert info['parameters'] == []
    assert 'source_redshift' in info['problems'][0]['message']


def test_a_lens_shows_its_critical_curves_and_caustics(client):
    # (of a singular isothermal sphere: the Einstein ring, of the Einstein
    # radius, whose caustic is a point at its centre; the field of the view
    # of its Einstein radius)
    view = client.post('/api/v1/views/lens', json=dict(
        project='study', lens=LENS, properties=LENS_PROPERTIES,
        size=96)).json()
    assert view['half'] == pytest.approx(2.5 * 1.2)
    assert len(view['alpha_x']) == 96 * 96
    ring = max(view['critical'], key=len)
    radii = [math.hypot(x, y) for x, y in ring]
    assert sum(radii) / len(radii) == pytest.approx(1.2, rel=0.02)
    caustic = view['caustics'][view['critical'].index(ring)]
    assert max(math.hypot(x, y) for x, y in caustic) < 0.05
    assert view['left_out'] == []
    response = client.post('/api/v1/views/lens', json=dict(
        project='study', lens=LENS, properties=dict(sie_theta_e=1),
        size=32))
    assert response.status_code in (400, 422)
