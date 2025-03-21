from . import utils
from .base import SpookBase
from .lin_solve import SpookLinSolve
from .quad_program import SpookL1, SpookPosL1, SpookPosL2
from .xval import XValidation

# from . __version__ import __version__
# from spook.vmi_special import PhotonFreqResVMI

SpookL2 = SpookLinSolve # alias for consistency with other solvers

__version__ = '1.0.1'
