"""
Tests for the services of the server: the workspace, the jobs (which run
the tasks of the library) and the live sessions.
"""

import asyncio

import numpy as np
import pytest

from gbkfit_server.services import frames
from gbkfit_server.services.errors import Invalid
from gbkfit_server.services.executors import LocalExecutor
from gbkfit_server.services.server import Server
from gbkfit_server.services.sessions import Latest


async def ended(job, timeout=60):
    """Wait for a job to end."""
    for _ in range(int(timeout / 0.1)):
        if job.state not in ('queued', 'running'):
            return job
        await asyncio.sleep(0.1)
    raise TimeoutError(f"job {job.id} is still {job.state}")


def test_the_workspace_has_projects_and_stays_in_its_folder(root):
    server = Server(root, LocalExecutor())
    assert server.workspace.projects() == ['study']
    project = server.workspace.project('study')
    assert project.pieces['setups'] == ['disk']
    assert project.pieces['observations'] == ['image']
    with pytest.raises(Invalid, match="out of the workspace"):
        server.workspace.read_text('../elsewhere.yaml')


def test_a_job_runs_a_setup_into_its_runs(root, setup):
    async def run():
        server = Server(root, LocalExecutor())
        states = []

        async def follow():
            async for event in server.events.subscribe():
                if event.kind == 'job':
                    states.append(event.data['state'])
        following = asyncio.create_task(follow())
        await asyncio.sleep(0)  # (it subscribes)
        options = dict(mode='model', config=setup, profile_iters=0)
        first = await ended(server.jobs.submit('eval', options))
        second = await ended(server.jobs.submit('eval', options))
        following.cancel()
        return server, first, second, states

    server, first, second, states = asyncio.run(run())
    assert (first.state, first.run) == ('complete', 'study/runs/disk')
    assert second.run.startswith('study/runs/disk_')
    assert states == ['queued', 'running', 'complete'] * 2
    run = server.workspace.run(first.run)
    assert run['status'] == 'complete'
    assert 'setup' in {p['kind'] for p in run['products']}


def test_a_job_of_a_configuration_out_of_the_setups_needs_a_folder(root):
    async def run():
        server = Server(root, LocalExecutor())
        job = server.jobs.submit('eval', dict(
            mode='model', config='study/params/disk.yaml', profile_iters=0))
        return await ended(job)
    job = asyncio.run(run())
    assert job.state == 'failed' and 'give the folder' in job.output


def test_the_latest_values_replace_those_not_taken():
    async def run():
        latest = Latest()
        for value in (1, 2, 3):
            latest.put(value)
        return await latest.get()
    assert asyncio.run(run()) == 3


def test_a_session_evaluates_its_model_at_new_values(root, setup):
    async def run():
        server = Server(root, LocalExecutor())
        session = await server.sessions.open(setup)
        results = [
            frames.decode(await session.evaluate(values))
            for values in ({}, dict(incl=80), dict(nothing=1))]
        await server.close()
        return session.info(), results

    info, (start, steeper, unknown) = asyncio.run(run())
    assert info['observations'] == [
        dict(label='observation0', items=['brightness'])]
    assert info['free']['incl'] == dict(value=45, min=0, max=90)
    (image,) = frames.arrays(*start)
    (steeper_image,) = frames.arrays(*steeper)
    assert image.shape == (12, 16) and image.sum() > 0
    assert not np.allclose(image, steeper_image)
    assert unknown[0] == dict(
        type='error', message="no free elements ['nothing']", nbytes=0)


def test_a_session_of_an_invalid_configuration_does_not_open(root):
    async def run():
        server = Server(root, LocalExecutor())
        await server.sessions.open('study/galaxies/disk.yaml')
    with pytest.raises(Invalid, match="cannot set up"):
        asyncio.run(run())
