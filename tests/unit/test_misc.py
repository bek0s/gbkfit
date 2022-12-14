#
# import numpy as np
# import pytest
#
# from gbkfit.params.pdescs import ParamDescDict, ParamScalarDesc, ParamVectorDesc
# from gbkfit.params.interpreter import Interpreter
#
#
#
# @pytest.fixture
# def fixture_desc_scalar():
#     return ParamScalarDesc('name', 'description', 10, -10, +np.inf)
#
#
# @pytest.fixture
# def fixture_desc_vector():
#     return ParamVectorDesc('name', 5, 'description', 10, -10, +np.inf)
#
#
# def test_interpreter():
#
#     # descs = dict(
#     #     a=ParamScalarDesc('a'),
#     #     b=ParamScalarDesc('b'),
#     #     c=ParamScalarDesc('c')
#     # )
#     #
#     # exprs = {
#     #     'b': 'c',
#     #     'c': 'a'
#     # }
#     #
#     # interpreter = Interpreter(descs, exprs)
#     #
#     # result = interpreter.evaluate({'a': 1})
#     #
#     # exit(result)
#
#
#
#     descs = dict(
#         a=ParamScalarDesc('a'),
#         b=ParamScalarDesc('b'),
#         c=ParamScalarDesc('c'),
#         d=ParamScalarDesc('d'),
#         e=ParamVectorDesc('e', 3),
#         f=ParamVectorDesc('f', 3),
#         g=ParamVectorDesc('g', 3),
#         h=ParamVectorDesc('h', 3),
#         i=ParamVectorDesc('i', 3),
#         j=ParamVectorDesc('j', 3)
#     )
#
#     exprs = {
#         'b': 'np.pi',
#         'c': 'np.cos(np.deg2rad(90))',
#         'd': 'a + b',
#         'e': '[1, a, a + b]',
#         'f': [1, 'a', 'a + b'],
#         'g': 'a',
#         'h': 'g',
#         'i': 'g + h',
#         'j[0]': 'j[2]',
#         'j[1]': 'a',
#         'j[2]': 'j[1]'
#     }
#
#     interpreter = Interpreter(descs, exprs)
#
#     result = interpreter.evaluate({'a': 1})
#
#
