"""
The renders of the server in their worker processes (see RenderPool): a
long render is announced and can be stopped, a waiting one is dropped
when its client goes, and the errors of a render are those of the render.
"""

import asyncio

import pytest

from gbkfit_server.services.errors import Cancelled, Invalid
from gbkfit_server.services.events import Events
from gbkfit_server.services.render_pool import RenderPool
from gbkfit_server.services.workspace import Workspace


# A thin gas disk, which renders at once, and a thick disk of many clouds,
# which renders for seconds or more
DISK = dict(components=[dict(
    type='thin_disk', loose=False, tilted=False, rnodes=list(range(0, 11, 2)),
    rptraits=dict(type='exponential'),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'), emission=dict(type='lines'))])
DISK_VALUES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=3, vpt_rt=2,
    vpt_vt=150, dpt_a=20)
SLOW = dict(components=[dict(
    type='thick_disk', loose=False, tilted=False, rnodes=list(range(0, 41, 2)),
    rptraits=dict(type='exponential'), rhtraits=dict(type='sech2', zmax=6),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'), emission=dict(type='lines'),
    clouds=dict(cflux=1e-6))])
SLOW_VALUES = dict(
    xpos=0, ypos=0, posa=0, incl=45, vsys=0, rpt_a=1, rpt_s=8, rht_s=1,
    vpt_rt=3, vpt_vt=200, dpt_a=20)


def render_args(galaxy, values):
    """The arguments of the render of a galaxy of the project 'study'."""
    return dict(project='study', options=galaxy, properties=values)


async def until(condition, seconds=60.0):
    """Wait until the condition holds (or fail after the seconds)."""
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the condition did not come to hold")


def test_a_long_render_is_announced_and_can_be_stopped(root):
    async def main():
        events = Events()
        seen = []

        async def listen():
            async for event in events.subscribe():
                seen.append(event)
        listener = asyncio.create_task(listen())
        pool = RenderPool(Workspace(root), events, size=1, moment=0.2)
        try:
            slow = asyncio.create_task(pool.render(
                'galaxy', render_args(SLOW, SLOW_VALUES), 'galaxy slow'))
            # (announced, and running: not waiting for its worker to start)
            await until(lambda: any(
                r['stage'] == 'evaluating' for r in pool.running()))
            info, = pool.running()
            assert (info['what'], info['kind']) == ('galaxy slow', 'galaxy')
            pool.cancel(info['id'])
            with pytest.raises(Cancelled):
                await slow
            assert pool.running() == []
            # (its worker was stopped: another renders the next)
            result = await pool.render(
                'galaxy', render_args(DISK, DISK_VALUES), 'galaxy disk')
            assert 'light' in result.maps
        finally:
            await pool.close()
            listener.cancel()
        states = [event.data['state'] for event in seen
                  if event.kind == 'render' and event.data['id'] == info['id']]
        assert states[0] == 'running' and states[-1] == 'cancelled'
    asyncio.run(main())


def test_a_waiting_render_is_dropped_when_its_client_goes(root):
    async def main():
        pool = RenderPool(Workspace(root), Events(), size=1, moment=60)
        try:
            slow = asyncio.create_task(pool.render(
                'galaxy', render_args(SLOW, SLOW_VALUES), 'galaxy slow'))
            await asyncio.sleep(0.5)

            async def gone():
                return True
            # (the one worker renders the slow one: this one waits, and its
            # client has gone)
            with pytest.raises(Cancelled):
                await pool.render(
                    'galaxy', render_args(DISK, DISK_VALUES), 'galaxy disk',
                    gone)
            assert not slow.done()
            slow.cancel()
        finally:
            await pool.close()
    asyncio.run(main())


def test_the_errors_of_a_render_are_its_own(root):
    async def main():
        pool = RenderPool(Workspace(root), Events())
        try:
            with pytest.raises(Invalid, match="component"):
                await pool.render(
                    'galaxy', render_args(dict(components=[]), {}),
                    'galaxy empty')
            # (the worker goes on)
            result = await pool.render(
                'galaxy', render_args(DISK, DISK_VALUES), 'galaxy disk')
            assert 'light' in result.maps
        finally:
            await pool.close()
    asyncio.run(main())
