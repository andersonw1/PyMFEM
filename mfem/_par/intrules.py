# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _intrules
else:
    import _intrules

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

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


import mfem._par.array
import mfem._par.mem_manager
class IntegrationPointArray(object):
    r"""Proxy of C++ mfem::Array< mfem::IntegrationPoint > class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(IntegrationPointArray self, int asize=0) -> IntegrationPointArray
        __init__(IntegrationPointArray self, IntegrationPoint _data, int asize) -> IntegrationPointArray
        __init__(IntegrationPointArray self, IntegrationPointArray src) -> IntegrationPointArray
        """
        _intrules.IntegrationPointArray_swiginit(self, _intrules.new_IntegrationPointArray(*args))

        if len(args) == 1 and isinstance(args[0], list):
            self.MakeDataOwner()



    __swig_destroy__ = _intrules.delete_IntegrationPointArray

    def GetData(self, *args):
        r"""
        GetData(IntegrationPointArray self) -> IntegrationPoint
        GetData(IntegrationPointArray self) -> IntegrationPoint
        """
        return _intrules.IntegrationPointArray_GetData(self, *args)

    def GetMemory(self, *args):
        r"""
        GetMemory(IntegrationPointArray self) -> mfem::Memory< mfem::IntegrationPoint >
        GetMemory(IntegrationPointArray self) -> mfem::Memory< mfem::IntegrationPoint > const &
        """
        return _intrules.IntegrationPointArray_GetMemory(self, *args)

    def UseDevice(self):
        r"""UseDevice(IntegrationPointArray self) -> bool"""
        return _intrules.IntegrationPointArray_UseDevice(self)

    def OwnsData(self):
        r"""OwnsData(IntegrationPointArray self) -> bool"""
        return _intrules.IntegrationPointArray_OwnsData(self)

    def StealData(self, p):
        r"""StealData(IntegrationPointArray self, mfem::IntegrationPoint ** p)"""
        return _intrules.IntegrationPointArray_StealData(self, p)

    def LoseData(self):
        r"""LoseData(IntegrationPointArray self)"""
        return _intrules.IntegrationPointArray_LoseData(self)

    def MakeDataOwner(self):
        r"""MakeDataOwner(IntegrationPointArray self)"""
        return _intrules.IntegrationPointArray_MakeDataOwner(self)

    def Size(self):
        r"""Size(IntegrationPointArray self) -> int"""
        return _intrules.IntegrationPointArray_Size(self)

    def SetSize(self, *args):
        r"""
        SetSize(IntegrationPointArray self, int nsize)
        SetSize(IntegrationPointArray self, int nsize, IntegrationPoint initval)
        SetSize(IntegrationPointArray self, int nsize, mfem::MemoryType mt)
        """
        return _intrules.IntegrationPointArray_SetSize(self, *args)

    def Capacity(self):
        r"""Capacity(IntegrationPointArray self) -> int"""
        return _intrules.IntegrationPointArray_Capacity(self)

    def Reserve(self, capacity):
        r"""Reserve(IntegrationPointArray self, int capacity)"""
        return _intrules.IntegrationPointArray_Reserve(self, capacity)

    def Append(self, *args):
        r"""
        Append(IntegrationPointArray self, IntegrationPoint el) -> int
        Append(IntegrationPointArray self, IntegrationPoint els, int nels) -> int
        Append(IntegrationPointArray self, IntegrationPointArray els) -> int
        """
        return _intrules.IntegrationPointArray_Append(self, *args)

    def Prepend(self, el):
        r"""Prepend(IntegrationPointArray self, IntegrationPoint el) -> int"""
        return _intrules.IntegrationPointArray_Prepend(self, el)

    def Last(self, *args):
        r"""
        Last(IntegrationPointArray self) -> IntegrationPoint
        Last(IntegrationPointArray self) -> IntegrationPoint
        """
        return _intrules.IntegrationPointArray_Last(self, *args)

    def DeleteLast(self):
        r"""DeleteLast(IntegrationPointArray self)"""
        return _intrules.IntegrationPointArray_DeleteLast(self)

    def DeleteAll(self):
        r"""DeleteAll(IntegrationPointArray self)"""
        return _intrules.IntegrationPointArray_DeleteAll(self)

    def Copy(self, copy):
        r"""Copy(IntegrationPointArray self, IntegrationPointArray copy)"""
        return _intrules.IntegrationPointArray_Copy(self, copy)

    def MakeRef(self, *args):
        r"""
        MakeRef(IntegrationPointArray self, IntegrationPoint arg2, int arg3)
        MakeRef(IntegrationPointArray self, IntegrationPointArray master)
        """
        return _intrules.IntegrationPointArray_MakeRef(self, *args)

    def GetSubArray(self, offset, sa_size, sa):
        r"""GetSubArray(IntegrationPointArray self, int offset, int sa_size, IntegrationPointArray sa)"""
        return _intrules.IntegrationPointArray_GetSubArray(self, offset, sa_size, sa)

    def begin(self):
        r"""begin(IntegrationPointArray self) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_begin(self)

    def end(self):
        r"""end(IntegrationPointArray self) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_end(self)

    def MemoryUsage(self):
        r"""MemoryUsage(IntegrationPointArray self) -> long"""
        return _intrules.IntegrationPointArray_MemoryUsage(self)

    def Read(self, on_dev=True):
        r"""Read(IntegrationPointArray self, bool on_dev=True) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_Read(self, on_dev)

    def HostRead(self):
        r"""HostRead(IntegrationPointArray self) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_HostRead(self)

    def Write(self, on_dev=True):
        r"""Write(IntegrationPointArray self, bool on_dev=True) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_Write(self, on_dev)

    def HostWrite(self):
        r"""HostWrite(IntegrationPointArray self) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_HostWrite(self)

    def ReadWrite(self, on_dev=True):
        r"""ReadWrite(IntegrationPointArray self, bool on_dev=True) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_ReadWrite(self, on_dev)

    def HostReadWrite(self):
        r"""HostReadWrite(IntegrationPointArray self) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray_HostReadWrite(self)

    def __setitem__(self, i, v):
        r"""__setitem__(IntegrationPointArray self, int i, IntegrationPoint v)"""
        return _intrules.IntegrationPointArray___setitem__(self, i, v)

    def __getitem__(self, i):
        r"""__getitem__(IntegrationPointArray self, int const i) -> IntegrationPoint"""
        return _intrules.IntegrationPointArray___getitem__(self, i)

    def Assign(self, *args):
        r"""
        Assign(IntegrationPointArray self, IntegrationPoint arg2)
        Assign(IntegrationPointArray self, IntegrationPoint a)
        """
        return _intrules.IntegrationPointArray_Assign(self, *args)

    def ToList(self):
        return [self[i] for i in range(self.Size())]



# Register IntegrationPointArray in _intrules:
_intrules.IntegrationPointArray_swigregister(IntegrationPointArray)

class IntegrationPoint(object):
    r"""Proxy of C++ mfem::IntegrationPoint class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    x = property(_intrules.IntegrationPoint_x_get, _intrules.IntegrationPoint_x_set, doc=r"""x : double""")
    y = property(_intrules.IntegrationPoint_y_get, _intrules.IntegrationPoint_y_set, doc=r"""y : double""")
    z = property(_intrules.IntegrationPoint_z_get, _intrules.IntegrationPoint_z_set, doc=r"""z : double""")
    weight = property(_intrules.IntegrationPoint_weight_get, _intrules.IntegrationPoint_weight_set, doc=r"""weight : double""")

    def Init(self):
        r"""Init(IntegrationPoint self)"""
        return _intrules.IntegrationPoint_Init(self)

    def Get(self, p, dim):
        r"""Get(IntegrationPoint self, double * p, int const dim)"""
        return _intrules.IntegrationPoint_Get(self, p, dim)

    def Set(self, *args):
        r"""
        Set(IntegrationPoint self, double const * p, int const dim)
        Set(IntegrationPoint self, double const x1, double const x2, double const x3, double const w)
        """
        return _intrules.IntegrationPoint_Set(self, *args)

    def Set3w(self, p):
        r"""Set3w(IntegrationPoint self, double const * p)"""
        return _intrules.IntegrationPoint_Set3w(self, p)

    def Set3(self, *args):
        r"""
        Set3(IntegrationPoint self, double const x1, double const x2, double const x3)
        Set3(IntegrationPoint self, double const * p)
        """
        return _intrules.IntegrationPoint_Set3(self, *args)

    def Set2w(self, *args):
        r"""
        Set2w(IntegrationPoint self, double const x1, double const x2, double const w)
        Set2w(IntegrationPoint self, double const * p)
        """
        return _intrules.IntegrationPoint_Set2w(self, *args)

    def Set2(self, *args):
        r"""
        Set2(IntegrationPoint self, double const x1, double const x2)
        Set2(IntegrationPoint self, double const * p)
        """
        return _intrules.IntegrationPoint_Set2(self, *args)

    def Set1w(self, *args):
        r"""
        Set1w(IntegrationPoint self, double const x1, double const w)
        Set1w(IntegrationPoint self, double const * p)
        """
        return _intrules.IntegrationPoint_Set1w(self, *args)

    def __init__(self):
        r"""__init__(IntegrationPoint self) -> IntegrationPoint"""
        _intrules.IntegrationPoint_swiginit(self, _intrules.new_IntegrationPoint())
    __swig_destroy__ = _intrules.delete_IntegrationPoint

# Register IntegrationPoint in _intrules:
_intrules.IntegrationPoint_swigregister(IntegrationPoint)

class IntegrationRule(IntegrationPointArray):
    r"""Proxy of C++ mfem::IntegrationRule class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(IntegrationRule self) -> IntegrationRule
        __init__(IntegrationRule self, int NP) -> IntegrationRule
        __init__(IntegrationRule self, IntegrationRule irx, IntegrationRule iry) -> IntegrationRule
        __init__(IntegrationRule self, IntegrationRule irx, IntegrationRule iry, IntegrationRule irz) -> IntegrationRule
        """
        _intrules.IntegrationRule_swiginit(self, _intrules.new_IntegrationRule(*args))

    def GetOrder(self):
        r"""GetOrder(IntegrationRule self) -> int"""
        return _intrules.IntegrationRule_GetOrder(self)

    def SetOrder(self, order):
        r"""SetOrder(IntegrationRule self, int const order)"""
        return _intrules.IntegrationRule_SetOrder(self, order)

    def GetNPoints(self):
        r"""GetNPoints(IntegrationRule self) -> int"""
        return _intrules.IntegrationRule_GetNPoints(self)

    def IntPoint(self, *args):
        r"""
        IntPoint(IntegrationRule self, int i) -> IntegrationPoint
        IntPoint(IntegrationRule self, int i) -> IntegrationPoint
        """
        return _intrules.IntegrationRule_IntPoint(self, *args)

    def GetWeights(self):
        r"""GetWeights(IntegrationRule self) -> doubleArray"""
        return _intrules.IntegrationRule_GetWeights(self)
    __swig_destroy__ = _intrules.delete_IntegrationRule

# Register IntegrationRule in _intrules:
_intrules.IntegrationRule_swigregister(IntegrationRule)

class QuadratureFunctions1D(object):
    r"""Proxy of C++ mfem::QuadratureFunctions1D class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def GaussLegendre(self, np, ir):
        r"""GaussLegendre(QuadratureFunctions1D self, int const np, IntegrationRule ir)"""
        return _intrules.QuadratureFunctions1D_GaussLegendre(self, np, ir)

    def GaussLobatto(self, np, ir):
        r"""GaussLobatto(QuadratureFunctions1D self, int const np, IntegrationRule ir)"""
        return _intrules.QuadratureFunctions1D_GaussLobatto(self, np, ir)

    def OpenUniform(self, np, ir):
        r"""OpenUniform(QuadratureFunctions1D self, int const np, IntegrationRule ir)"""
        return _intrules.QuadratureFunctions1D_OpenUniform(self, np, ir)

    def ClosedUniform(self, np, ir):
        r"""ClosedUniform(QuadratureFunctions1D self, int const np, IntegrationRule ir)"""
        return _intrules.QuadratureFunctions1D_ClosedUniform(self, np, ir)

    def OpenHalfUniform(self, np, ir):
        r"""OpenHalfUniform(QuadratureFunctions1D self, int const np, IntegrationRule ir)"""
        return _intrules.QuadratureFunctions1D_OpenHalfUniform(self, np, ir)

    def GivePolyPoints(self, np, pts, type):
        r"""GivePolyPoints(QuadratureFunctions1D self, int const np, double * pts, int const type)"""
        return _intrules.QuadratureFunctions1D_GivePolyPoints(self, np, pts, type)

    def __init__(self):
        r"""__init__(QuadratureFunctions1D self) -> QuadratureFunctions1D"""
        _intrules.QuadratureFunctions1D_swiginit(self, _intrules.new_QuadratureFunctions1D())
    __swig_destroy__ = _intrules.delete_QuadratureFunctions1D

# Register QuadratureFunctions1D in _intrules:
_intrules.QuadratureFunctions1D_swigregister(QuadratureFunctions1D)

class Quadrature1D(object):
    r"""Proxy of C++ mfem::Quadrature1D class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    Invalid = _intrules.Quadrature1D_Invalid
    
    GaussLegendre = _intrules.Quadrature1D_GaussLegendre
    
    GaussLobatto = _intrules.Quadrature1D_GaussLobatto
    
    OpenUniform = _intrules.Quadrature1D_OpenUniform
    
    ClosedUniform = _intrules.Quadrature1D_ClosedUniform
    
    OpenHalfUniform = _intrules.Quadrature1D_OpenHalfUniform
    

    @staticmethod
    def CheckClosed(type):
        r"""CheckClosed(int type) -> int"""
        return _intrules.Quadrature1D_CheckClosed(type)

    @staticmethod
    def CheckOpen(type):
        r"""CheckOpen(int type) -> int"""
        return _intrules.Quadrature1D_CheckOpen(type)

    def __init__(self):
        r"""__init__(Quadrature1D self) -> Quadrature1D"""
        _intrules.Quadrature1D_swiginit(self, _intrules.new_Quadrature1D())
    __swig_destroy__ = _intrules.delete_Quadrature1D

# Register Quadrature1D in _intrules:
_intrules.Quadrature1D_swigregister(Quadrature1D)

def Quadrature1D_CheckClosed(type):
    r"""Quadrature1D_CheckClosed(int type) -> int"""
    return _intrules.Quadrature1D_CheckClosed(type)

def Quadrature1D_CheckOpen(type):
    r"""Quadrature1D_CheckOpen(int type) -> int"""
    return _intrules.Quadrature1D_CheckOpen(type)

class IntegrationRules(object):
    r"""Proxy of C++ mfem::IntegrationRules class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""__init__(IntegrationRules self, int Ref=0, int type=GaussLegendre) -> IntegrationRules"""
        _intrules.IntegrationRules_swiginit(self, _intrules.new_IntegrationRules(*args))

    def Get(self, GeomType, Order):
        r"""Get(IntegrationRules self, int GeomType, int Order) -> IntegrationRule"""
        return _intrules.IntegrationRules_Get(self, GeomType, Order)

    def Set(self, GeomType, Order, IntRule):
        r"""Set(IntegrationRules self, int GeomType, int Order, IntegrationRule IntRule)"""
        return _intrules.IntegrationRules_Set(self, GeomType, Order, IntRule)

    def SetOwnRules(self, o):
        r"""SetOwnRules(IntegrationRules self, int o)"""
        return _intrules.IntegrationRules_SetOwnRules(self, o)
    __swig_destroy__ = _intrules.delete_IntegrationRules

# Register IntegrationRules in _intrules:
_intrules.IntegrationRules_swigregister(IntegrationRules)


cvar = _intrules.cvar
IntRules = cvar.IntRules
RefinedIntRules = cvar.RefinedIntRules

