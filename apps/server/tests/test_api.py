"""
Tests for the API of the server: its token, and the round trips of a
project, a job and its run, and a live session.
"""

import json
import pathlib
import time

import pytest
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
    # A setup with an unknown section, and a piece with a misspelt type
    # (in the setup that names it)
    text = (pathlib.Path(root) / setup).read_text() + 'bogus: 1\n'
    (diagnostic,) = client.post('/api/v1/check', json=dict(
        path=setup, text=text)).json()['diagnostics']
    assert diagnostic['severity'] == 'warning'
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
    assert 'smdisk' in client.get(f'/api/v1/files/{path}').text
    assert client.post('/api/v1/pieces', json=dict(
        project='new', folder='galaxies', name='disk')).status_code == 400
    assert sorted(client.get('/api/v1/projects').json()) == ['new', 'study']
