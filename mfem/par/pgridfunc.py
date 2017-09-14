# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_pgridfunc', [dirname(__file__)])
        except ImportError:
            import _pgridfunc
            return _pgridfunc
        if fp is not None:
            try:
                _mod = imp.load_module('_pgridfunc', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _pgridfunc = swig_import_helper()
    del swig_import_helper
else:
    import _pgridfunc
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


try:
    import weakref
    weakref_proxy = weakref.proxy
except Exception:
    weakref_proxy = lambda x: x



_pgridfunc.MFEM_TIMER_TYPE_swigconstant(_pgridfunc)
MFEM_TIMER_TYPE = _pgridfunc.MFEM_TIMER_TYPE
import pfespace
import operators
import vector
import array
import ostream_typemap
import fespace
import coefficient
import matrix
import intrules
import sparsemat
import densemat
import eltrans
import fe
import mesh
import ncmesh
import element
import geom
import table
import vertex
import gridfunc
import bilininteg
import fe_coll
import lininteg
import linearform
import pmesh
import pncmesh
import communication
import sets
import hypre

def GlobalLpNorm(p, loc_norm, comm):
    return _pgridfunc.GlobalLpNorm(p, loc_norm, comm)
GlobalLpNorm = _pgridfunc.GlobalLpNorm
class ParGridFunction(gridfunc.GridFunction):
    __swig_setmethods__ = {}
    for _s in [gridfunc.GridFunction]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, ParGridFunction, name, value)
    __swig_getmethods__ = {}
    for _s in [gridfunc.GridFunction]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, ParGridFunction, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _pgridfunc.new_ParGridFunction(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this

    def ParFESpace(self):
        return _pgridfunc.ParGridFunction_ParFESpace(self)

    def Update(self):
        return _pgridfunc.ParGridFunction_Update(self)

    def SetSpace(self, f):
        return _pgridfunc.ParGridFunction_SetSpace(self, f)

    def MakeRef(self, *args):
        return _pgridfunc.ParGridFunction_MakeRef(self, *args)

    def Distribute(self, *args):
        return _pgridfunc.ParGridFunction_Distribute(self, *args)

    def AddDistribute(self, *args):
        return _pgridfunc.ParGridFunction_AddDistribute(self, *args)

    def SetFromTrueDofs(self, tv):
        return _pgridfunc.ParGridFunction_SetFromTrueDofs(self, tv)

    def Assign(self, *args):
        return _pgridfunc.ParGridFunction_Assign(self, *args)

    def GetTrueDofs(self, *args):
        return _pgridfunc.ParGridFunction_GetTrueDofs(self, *args)

    def ParallelAverage(self, *args):
        return _pgridfunc.ParGridFunction_ParallelAverage(self, *args)

    def ParallelProject(self, *args):
        return _pgridfunc.ParGridFunction_ParallelProject(self, *args)

    def ParallelAssemble(self, *args):
        return _pgridfunc.ParGridFunction_ParallelAssemble(self, *args)

    def ExchangeFaceNbrData(self):
        return _pgridfunc.ParGridFunction_ExchangeFaceNbrData(self)

    def FaceNbrData(self, *args):
        return _pgridfunc.ParGridFunction_FaceNbrData(self, *args)

    def GetValue(self, *args):
        return _pgridfunc.ParGridFunction_GetValue(self, *args)

    def ProjectCoefficient(self, *args):
        return _pgridfunc.ParGridFunction_ProjectCoefficient(self, *args)

    def ProjectDiscCoefficient(self, *args):
        return _pgridfunc.ParGridFunction_ProjectDiscCoefficient(self, *args)

    def ComputeL1Error(self, *args):
        return _pgridfunc.ParGridFunction_ComputeL1Error(self, *args)

    def ComputeL2Error(self, *args):
        return _pgridfunc.ParGridFunction_ComputeL2Error(self, *args)

    def ComputeMaxError(self, *args):
        return _pgridfunc.ParGridFunction_ComputeMaxError(self, *args)

    def ComputeLpError(self, *args):
        return _pgridfunc.ParGridFunction_ComputeLpError(self, *args)

    def ComputeFlux(self, blfi, flux, wcoef=1, subdomain=-1):
        return _pgridfunc.ParGridFunction_ComputeFlux(self, blfi, flux, wcoef, subdomain)

    def Save(self, out):
        return _pgridfunc.ParGridFunction_Save(self, out)

    def SaveAsOne(self, *args):
        return _pgridfunc.ParGridFunction_SaveAsOne(self, *args)
    __swig_destroy__ = _pgridfunc.delete_ParGridFunction
    __del__ = lambda self: None
ParGridFunction_swigregister = _pgridfunc.ParGridFunction_swigregister
ParGridFunction_swigregister(ParGridFunction)


def L2ZZErrorEstimator(flux_integrator, x, smooth_flux_fes, flux_fes, errors, norm_p=2, solver_tol=1e-12, solver_max_it=200):
    return _pgridfunc.L2ZZErrorEstimator(flux_integrator, x, smooth_flux_fes, flux_fes, errors, norm_p, solver_tol, solver_max_it)
L2ZZErrorEstimator = _pgridfunc.L2ZZErrorEstimator
# This file is compatible with both classic and new-style classes.


