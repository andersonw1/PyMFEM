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
    from . import _point
else:
    import _point

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


import weakref

import mfem._ser.element
import mfem._ser.array
import mfem._ser.mem_manager
import mfem._ser.densemat
import mfem._ser.vector
import mfem._ser.operators
import mfem._ser.matrix
import mfem._ser.geom
import mfem._ser.intrules
import mfem._ser.table
import mfem._ser.hash
class Point(mfem._ser.element.Element):
    r"""Proxy of C++ mfem::Point class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(Point self) -> Point
        __init__(Point self, int const * ind, int attr=-1) -> Point
        """
        _point.Point_swiginit(self, _point.new_Point(*args))

    def GetType(self):
        r"""GetType(Point self) -> mfem::Element::Type"""
        return _point.Point_GetType(self)

    def GetVertices(self, *args):
        r"""
        GetVertices(Point self, intArray v)
        GetVertices(Point self) -> int *
        """
        return _point.Point_GetVertices(self, *args)

    def GetNVertices(self):
        r"""GetNVertices(Point self) -> int"""
        return _point.Point_GetNVertices(self)

    def GetNEdges(self):
        r"""GetNEdges(Point self) -> int"""
        return _point.Point_GetNEdges(self)

    def GetEdgeVertices(self, ei):
        r"""GetEdgeVertices(Point self, int ei) -> int const *"""
        return _point.Point_GetEdgeVertices(self, ei)

    def GetNFaces(self, nFaceVertices):
        r"""GetNFaces(Point self, int & nFaceVertices) -> int"""
        return _point.Point_GetNFaces(self, nFaceVertices)

    def GetFaceVertices(self, fi):
        r"""GetFaceVertices(Point self, int fi) -> int const *"""
        return _point.Point_GetFaceVertices(self, fi)

    def Duplicate(self, m):
        r"""Duplicate(Point self, mfem::Mesh * m) -> Element"""
        return _point.Point_Duplicate(self, m)
    __swig_destroy__ = _point.delete_Point

# Register Point in _point:
_point.Point_swigregister(Point)


cvar = _point.cvar

