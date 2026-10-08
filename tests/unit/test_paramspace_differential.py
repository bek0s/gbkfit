"""
Temporary: compare the new parameter engine (ParamSpace) with the old one
on random parameter properties. Deleted together with the old engine.
"""

import logging

import numpy as np
import pytest

from gbkfit.params.space import ParamSpace

from test_paramspace import load_eval_params, load_fit_params, make_pdescs


def random_expression(rng, pdescs, target):
    """
    A random expression of one or two parameters (or elements). It reads
    the parameter it sets (target) only sometimes, because that is
    usually a cycle.
    """
    others = [name for name in pdescs if name != target]
    terms = [str(rng.integers(1, 5))]
    for _ in range(rng.integers(1, 3)):
        name = rng.choice(
            others if others and rng.random() < 0.9 else list(pdescs))
        size = pdescs[name].size()
        if pdescs[name].type() == 'scalar':
            terms.append(name)
        elif rng.random() < 0.5:
            index = int(rng.integers(-size, size))
            terms.append(f'{name}[{index}]')
        else:
            start = int(rng.integers(0, size))
            terms.append(f'np.sum({name}[{start}:])')
    operators = rng.choice(['+', '-', '*'], len(terms) - 1)
    expression = terms[0]
    for operator, term in zip(operators, terms[1:]):
        expression += f' {operator} {term}'
    return expression


def random_value(rng, pdescs, target, fit):
    kind = rng.choice(['number', 'expression', 'free'] if fit
                      else ['number', 'expression'])
    if kind == 'number':
        return float(rng.integers(-5, 6))
    if kind == 'expression':
        return random_expression(rng, pdescs, target)
    return {'value': float(rng.integers(-5, 6)), 'min': -10}


def random_keys(rng, name, size):
    """Keys that select every element of a vector exactly once."""
    order = rng.permutation(size)
    keys = []
    while len(order):
        count = int(rng.integers(1, len(order) + 1))
        chunk, order = sorted(order[:count]), order[count:]
        if len(chunk) == 1:
            index = chunk[0] - size if rng.random() < 0.5 else chunk[0]
            keys.append((f'{name}[{index}]', 1, True))
        elif chunk == list(range(chunk[0], chunk[-1] + 1)):
            keys.append((f'{name}[{chunk[0]}:{chunk[-1] + 1}]', len(chunk),
                         False))
        else:
            indices = ', '.join(str(i) for i in chunk)
            keys.append((f'{name}[[{indices}]]', len(chunk), False))
    return keys


def random_case(rng, fit):
    sizes = {f's{i}': None for i in range(rng.integers(1, 4))}
    sizes |= {f'v{i}': int(rng.integers(1, 6))
              for i in range(rng.integers(1, 3))}
    pdescs = make_pdescs(**sizes)
    properties = {}
    for name, size in sizes.items():
        mode = 'whole' if size is None else rng.choice(
            ['whole', 'list', 'keys'])
        if mode == 'whole':
            properties[name] = random_value(rng, pdescs, name, fit)
        elif mode == 'list':
            properties[name] = [
                random_value(rng, pdescs, name, fit) for _ in range(size)]
        else:
            for key, count, element in random_keys(rng, name, size):
                as_list = not element and rng.random() < 0.3
                properties[key] = (
                    [random_value(rng, pdescs, name, fit)
                     for _ in range(count)]
                    if as_list else random_value(rng, pdescs, name, fit))
    # Sometimes, a mistake: a missing or a repeated element
    if rng.random() < 0.1:
        del properties[rng.choice(list(properties))]
    if rng.random() < 0.1:
        name = rng.choice(list(sizes))
        properties[name if sizes[name] is None else f'{name}[0]'] = 1.0
    return pdescs, properties


def expand_vector_expressions(pdescs, properties):
    """
    The old engine sets a vector given a scalar expression (e.g. 'v':
    'a * 2') to a scalar until all expressions are evaluated, so other
    expressions cannot read its elements (a bug). The same expression
    for each element works.
    """
    return {
        key: [value] * pdescs[key].size()
        if isinstance(value, str) and key in pdescs
        and pdescs[key].type() == 'vector' else value
        for key, value in properties.items()}


def old_result(pdescs, properties, fit, rng):
    if fit:
        params = load_fit_params(pdescs, properties)
        names = [params.exploded_names(*kinds) for kinds in (
            (False, False, True), (False, True, False), (True, False, False))]
        free = {n: float(rng.integers(-5, 6)) for n in names[0]}
        return names, free, params.evaluate(free)
    params = load_eval_params(pdescs, properties)
    names = [[], params.exploded_names(False, True),
             params.exploded_names(True, False)]
    return names, {}, params.evaluate()


def new_result(pdescs, properties, free):
    space = ParamSpace(pdescs, properties)
    names = [space.names(*kinds) for kinds in (
        (True, False, False), (False, True, False), (False, False, True))]
    return names, space.evaluate(free)


@pytest.mark.parametrize('fit', [False, True])
def test_same_as_old_engine(fit):
    logging.disable(logging.WARNING)
    rng = np.random.default_rng(1)
    counts = dict(errors=0, cases=0, old_bug=0)
    for _ in range(1000):
        pdescs, properties = random_case(rng, fit)
        state = rng.bit_generator.state
        try:
            old_names, free, old_values = old_result(
                pdescs, properties, fit, rng)
        except Exception:
            rng.bit_generator.state = state
            try:
                old_names, free, old_values = old_result(
                    pdescs, expand_vector_expressions(pdescs, properties),
                    fit, rng)
                counts['old_bug'] += 1
            except Exception:
                counts['errors'] += 1
                with pytest.raises(Exception):
                    space = ParamSpace(pdescs, properties)
                    space.evaluate(dict.fromkeys(space.names(
                        free=True, tied=False, fixed=False), 1.0))
                continue
        counts['cases'] += 1
        new_names, new_values = new_result(pdescs, properties, free)
        assert new_names == old_names, properties
        for name in pdescs:
            np.testing.assert_allclose(
                new_values[name], old_values[name], err_msg=str(properties))
    logging.disable(logging.NOTSET)
    # Both kinds of case must be common
    assert counts['errors'] > 100 and counts['cases'] > 300, counts
