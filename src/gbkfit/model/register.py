
def _register_dmodels():
    from .dmodel import parser
    from .dmodels import DModelImage, DModelLSlit, DModelMMaps, DModelSCube
    parser.register(DModelImage)
    parser.register(DModelLSlit)
    parser.register(DModelMMaps)
    parser.register(DModelSCube)


_register_dmodels()
