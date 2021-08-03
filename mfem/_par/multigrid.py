# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _multigrid
else:
    import _multigrid

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

_swig_new_instance_method = _multigrid.SWIG_PyInstanceMethod_New
_swig_new_static_method = _multigrid.SWIG_PyStaticMethod_New

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


import weakref

import mfem._par.array
import mfem._par.mem_manager
import mfem._par.vector
import mfem._par.bilinearform
import mfem._par.globals
import mfem._par.fespace
import mfem._par.coefficient
import mfem._par.matrix
import mfem._par.operators
import mfem._par.intrules
import mfem._par.sparsemat
import mfem._par.densemat
import mfem._par.eltrans
import mfem._par.fe
import mfem._par.geom
import mfem._par.mesh
import mfem._par.sort_pairs
import mfem._par.ncmesh
import mfem._par.vtk
import mfem._par.element
import mfem._par.table
import mfem._par.hash
import mfem._par.vertex
import mfem._par.gridfunc
import mfem._par.bilininteg
import mfem._par.fe_coll
import mfem._par.lininteg
import mfem._par.linearform
import mfem._par.nonlininteg
import mfem._par.handle
import mfem._par.hypre
import mfem._par.restriction
import mfem._par.fespacehierarchy
class Multigrid(mfem._par.operators.Solver):
    r"""Proxy of C++ mfem::Multigrid class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    CycleType_VCYCLE = _multigrid.Multigrid_CycleType_VCYCLE
    
    CycleType_WCYCLE = _multigrid.Multigrid_CycleType_WCYCLE
    

    def __init__(self, *args):
        r"""
        __init__(Multigrid self) -> Multigrid
        __init__(Multigrid self, mfem::Array< mfem::Operator * > const & operators_, mfem::Array< mfem::Solver * > const & smoothers_, mfem::Array< mfem::Operator * > const & prolongations_, mfem::Array< bool > const & ownedOperators_, mfem::Array< bool > const & ownedSmoothers_, mfem::Array< bool > const & ownedProlongations_) -> Multigrid
        """
        _multigrid.Multigrid_swiginit(self, _multigrid.new_Multigrid(*args))
    __swig_destroy__ = _multigrid.delete_Multigrid

    def AddLevel(self, opr, smoother, ownOperator, ownSmoother):
        r"""AddLevel(Multigrid self, Operator opr, Solver smoother, bool ownOperator, bool ownSmoother)"""
        return _multigrid.Multigrid_AddLevel(self, opr, smoother, ownOperator, ownSmoother)
    AddLevel = _swig_new_instance_method(_multigrid.Multigrid_AddLevel)

    def NumLevels(self):
        r"""NumLevels(Multigrid self) -> int"""
        return _multigrid.Multigrid_NumLevels(self)
    NumLevels = _swig_new_instance_method(_multigrid.Multigrid_NumLevels)

    def GetFinestLevelIndex(self):
        r"""GetFinestLevelIndex(Multigrid self) -> int"""
        return _multigrid.Multigrid_GetFinestLevelIndex(self)
    GetFinestLevelIndex = _swig_new_instance_method(_multigrid.Multigrid_GetFinestLevelIndex)

    def GetOperatorAtLevel(self, *args):
        r"""
        GetOperatorAtLevel(Multigrid self, int level) -> Operator
        GetOperatorAtLevel(Multigrid self, int level) -> Operator
        """
        return _multigrid.Multigrid_GetOperatorAtLevel(self, *args)
    GetOperatorAtLevel = _swig_new_instance_method(_multigrid.Multigrid_GetOperatorAtLevel)

    def GetOperatorAtFinestLevel(self, *args):
        r"""
        GetOperatorAtFinestLevel(Multigrid self) -> Operator
        GetOperatorAtFinestLevel(Multigrid self) -> Operator
        """
        return _multigrid.Multigrid_GetOperatorAtFinestLevel(self, *args)
    GetOperatorAtFinestLevel = _swig_new_instance_method(_multigrid.Multigrid_GetOperatorAtFinestLevel)

    def GetSmootherAtLevel(self, *args):
        r"""
        GetSmootherAtLevel(Multigrid self, int level) -> Solver
        GetSmootherAtLevel(Multigrid self, int level) -> Solver
        """
        return _multigrid.Multigrid_GetSmootherAtLevel(self, *args)
    GetSmootherAtLevel = _swig_new_instance_method(_multigrid.Multigrid_GetSmootherAtLevel)

    def SetCycleType(self, cycleType_, preSmoothingSteps_, postSmoothingSteps_):
        r"""SetCycleType(Multigrid self, mfem::Multigrid::CycleType cycleType_, int preSmoothingSteps_, int postSmoothingSteps_)"""
        return _multigrid.Multigrid_SetCycleType(self, cycleType_, preSmoothingSteps_, postSmoothingSteps_)
    SetCycleType = _swig_new_instance_method(_multigrid.Multigrid_SetCycleType)

    def Mult(self, x, y):
        r"""Mult(Multigrid self, Vector x, Vector y)"""
        return _multigrid.Multigrid_Mult(self, x, y)
    Mult = _swig_new_instance_method(_multigrid.Multigrid_Mult)

    def SetOperator(self, op):
        r"""SetOperator(Multigrid self, Operator op)"""
        return _multigrid.Multigrid_SetOperator(self, op)
    SetOperator = _swig_new_instance_method(_multigrid.Multigrid_SetOperator)

# Register Multigrid in _multigrid:
_multigrid.Multigrid_swigregister(Multigrid)

class GeometricMultigrid(Multigrid):
    r"""Proxy of C++ mfem::GeometricMultigrid class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, fespaces_):
        r"""__init__(GeometricMultigrid self, FiniteElementSpaceHierarchy fespaces_) -> GeometricMultigrid"""
        _multigrid.GeometricMultigrid_swiginit(self, _multigrid.new_GeometricMultigrid(fespaces_))
    __swig_destroy__ = _multigrid.delete_GeometricMultigrid

    def FormFineLinearSystem(self, x, b, A, X, B):
        r"""FormFineLinearSystem(GeometricMultigrid self, Vector x, Vector b, OperatorHandle A, Vector X, Vector B)"""
        return _multigrid.GeometricMultigrid_FormFineLinearSystem(self, x, b, A, X, B)
    FormFineLinearSystem = _swig_new_instance_method(_multigrid.GeometricMultigrid_FormFineLinearSystem)

    def RecoverFineFEMSolution(self, X, b, x):
        r"""RecoverFineFEMSolution(GeometricMultigrid self, Vector X, Vector b, Vector x)"""
        return _multigrid.GeometricMultigrid_RecoverFineFEMSolution(self, X, b, x)
    RecoverFineFEMSolution = _swig_new_instance_method(_multigrid.GeometricMultigrid_RecoverFineFEMSolution)

# Register GeometricMultigrid in _multigrid:
_multigrid.GeometricMultigrid_swigregister(GeometricMultigrid)

class PyGeometricMultigrid(GeometricMultigrid):
    r"""Proxy of C++ PyGeometricMultigrid class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, fespaces_):
        r"""__init__(PyGeometricMultigrid self, FiniteElementSpaceHierarchy fespaces_) -> PyGeometricMultigrid"""
        _multigrid.PyGeometricMultigrid_swiginit(self, _multigrid.new_PyGeometricMultigrid(fespaces_))

    def AppendBilinearForm(self, form):
        r"""AppendBilinearForm(PyGeometricMultigrid self, BilinearForm form)"""

        if not hasattr(self, "_forms"): self._forms = []
        self._forms.append(form)
        form.thisown = 0


        return _multigrid.PyGeometricMultigrid_AppendBilinearForm(self, form)


    def AppendEssentialTDofs(self, ess):
        r"""AppendEssentialTDofs(PyGeometricMultigrid self, intArray ess)"""

        if not hasattr(self, "_esss"): self._esss = []
        self._esss.append(ess)	    
        ess.thisown = 0


        return _multigrid.PyGeometricMultigrid_AppendEssentialTDofs(self, ess)


    @property						     
    def bfs(self):
       return self._forms



    @property						     
    def essentialTrueDofs(self):
       return self._esss


    __swig_destroy__ = _multigrid.delete_PyGeometricMultigrid

# Register PyGeometricMultigrid in _multigrid:
_multigrid.PyGeometricMultigrid_swigregister(PyGeometricMultigrid)



