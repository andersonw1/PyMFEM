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
    from . import _array
else:
    import _array

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


import mfem._ser.mem_manager
class intArray(object):
    r"""Proxy of C++ mfem::Array< int > class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(intArray self, int asize=0) -> intArray
        __init__(intArray self, int * _data) -> intArray
        __init__(intArray self, intArray src) -> intArray
        """
        _array.intArray_swiginit(self, _array.new_intArray(*args))

        if len(args) == 1 and isinstance(args[0], list):
            self.MakeDataOwner()



    __swig_destroy__ = _array.delete_intArray

    def GetData(self, *args):
        r"""
        GetData(intArray self) -> int
        GetData(intArray self) -> int const *
        """
        return _array.intArray_GetData(self, *args)

    def GetMemory(self, *args):
        r"""
        GetMemory(intArray self) -> mfem::Memory< int >
        GetMemory(intArray self) -> mfem::Memory< int > const &
        """
        return _array.intArray_GetMemory(self, *args)

    def UseDevice(self):
        r"""UseDevice(intArray self) -> bool"""
        return _array.intArray_UseDevice(self)

    def OwnsData(self):
        r"""OwnsData(intArray self) -> bool"""
        return _array.intArray_OwnsData(self)

    def StealData(self, p):
        r"""StealData(intArray self, int ** p)"""
        return _array.intArray_StealData(self, p)

    def LoseData(self):
        r"""LoseData(intArray self)"""
        return _array.intArray_LoseData(self)

    def MakeDataOwner(self):
        r"""MakeDataOwner(intArray self)"""
        return _array.intArray_MakeDataOwner(self)

    def Size(self):
        r"""Size(intArray self) -> int"""
        return _array.intArray_Size(self)

    def SetSize(self, *args):
        r"""
        SetSize(intArray self, int nsize)
        SetSize(intArray self, int nsize, int const & initval)
        SetSize(intArray self, int nsize, mfem::MemoryType mt)
        """
        return _array.intArray_SetSize(self, *args)

    def Capacity(self):
        r"""Capacity(intArray self) -> int"""
        return _array.intArray_Capacity(self)

    def Reserve(self, capacity):
        r"""Reserve(intArray self, int capacity)"""
        return _array.intArray_Reserve(self, capacity)

    def Append(self, *args):
        r"""
        Append(intArray self, int const & el) -> int
        Append(intArray self, int const * els, int nels) -> int
        Append(intArray self, intArray els) -> int
        """
        return _array.intArray_Append(self, *args)

    def Prepend(self, el):
        r"""Prepend(intArray self, int const & el) -> int"""
        return _array.intArray_Prepend(self, el)

    def Last(self, *args):
        r"""
        Last(intArray self) -> int
        Last(intArray self) -> int const &
        """
        return _array.intArray_Last(self, *args)

    def Union(self, el):
        r"""Union(intArray self, int const & el) -> int"""
        return _array.intArray_Union(self, el)

    def Find(self, el):
        r"""Find(intArray self, int const & el) -> int"""
        return _array.intArray_Find(self, el)

    def FindSorted(self, el):
        r"""FindSorted(intArray self, int const & el) -> int"""
        return _array.intArray_FindSorted(self, el)

    def DeleteLast(self):
        r"""DeleteLast(intArray self)"""
        return _array.intArray_DeleteLast(self)

    def DeleteFirst(self, el):
        r"""DeleteFirst(intArray self, int const & el)"""
        return _array.intArray_DeleteFirst(self, el)

    def DeleteAll(self):
        r"""DeleteAll(intArray self)"""
        return _array.intArray_DeleteAll(self)

    def Copy(self, copy):
        r"""Copy(intArray self, intArray copy)"""
        return _array.intArray_Copy(self, copy)

    def MakeRef(self, *args):
        r"""
        MakeRef(intArray self, int * arg2, int arg3)
        MakeRef(intArray self, intArray master)
        """
        return _array.intArray_MakeRef(self, *args)

    def GetSubArray(self, offset, sa_size, sa):
        r"""GetSubArray(intArray self, int offset, int sa_size, intArray sa)"""
        return _array.intArray_GetSubArray(self, offset, sa_size, sa)

    def Load(self, *args):
        r"""
        Load(intArray self, std::istream & _in, int fmt=0)
        Load(intArray self, int new_size, std::istream & _in)
        """
        return _array.intArray_Load(self, *args)

    def Max(self):
        r"""Max(intArray self) -> int"""
        return _array.intArray_Max(self)

    def Min(self):
        r"""Min(intArray self) -> int"""
        return _array.intArray_Min(self)

    def Sort(self):
        r"""Sort(intArray self)"""
        return _array.intArray_Sort(self)

    def Unique(self):
        r"""Unique(intArray self)"""
        return _array.intArray_Unique(self)

    def IsSorted(self):
        r"""IsSorted(intArray self) -> int"""
        return _array.intArray_IsSorted(self)

    def PartialSum(self):
        r"""PartialSum(intArray self)"""
        return _array.intArray_PartialSum(self)

    def Sum(self):
        r"""Sum(intArray self) -> int"""
        return _array.intArray_Sum(self)

    def begin(self):
        r"""begin(intArray self) -> int *"""
        return _array.intArray_begin(self)

    def end(self):
        r"""end(intArray self) -> int *"""
        return _array.intArray_end(self)

    def MemoryUsage(self):
        r"""MemoryUsage(intArray self) -> long"""
        return _array.intArray_MemoryUsage(self)

    def Read(self, on_dev=True):
        r"""Read(intArray self, bool on_dev=True) -> int const"""
        return _array.intArray_Read(self, on_dev)

    def HostRead(self):
        r"""HostRead(intArray self) -> int const *"""
        return _array.intArray_HostRead(self)

    def Write(self, on_dev=True):
        r"""Write(intArray self, bool on_dev=True) -> int"""
        return _array.intArray_Write(self, on_dev)

    def HostWrite(self):
        r"""HostWrite(intArray self) -> int *"""
        return _array.intArray_HostWrite(self)

    def ReadWrite(self, on_dev=True):
        r"""ReadWrite(intArray self, bool on_dev=True) -> int"""
        return _array.intArray_ReadWrite(self, on_dev)

    def HostReadWrite(self):
        r"""HostReadWrite(intArray self) -> int *"""
        return _array.intArray_HostReadWrite(self)

    def __setitem__(self, i, v):
        r"""__setitem__(intArray self, int i, int const v)"""
        return _array.intArray___setitem__(self, i, v)

    def __getitem__(self, i):
        r"""__getitem__(intArray self, int const i) -> int const &"""
        return _array.intArray___getitem__(self, i)

    def Assign(self, *args):
        r"""
        Assign(intArray self, int const * arg2)
        Assign(intArray self, int const & a)
        """
        return _array.intArray_Assign(self, *args)

    def ToList(self):
        return [self[i] for i in range(self.Size())]



    def Print(self, *args):
        r"""
        Print(intArray self, std::ostream & out=mfem::out, int width=4)
        Print(intArray self, char const * file, int precision=8)
        """
        return _array.intArray_Print(self, *args)

    def Save(self, *args):
        r"""
        Save(intArray self, std::ostream & out, int fmt=0)
        Save(intArray self, char const * file, int precision=8)
        Save(intArray self)
        """
        return _array.intArray_Save(self, *args)

# Register intArray in _array:
_array.intArray_swigregister(intArray)

class doubleArray(object):
    r"""Proxy of C++ mfem::Array< double > class."""

    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        r"""
        __init__(doubleArray self, int asize=0) -> doubleArray
        __init__(doubleArray self, double * _data) -> doubleArray
        __init__(doubleArray self, doubleArray src) -> doubleArray
        """
        _array.doubleArray_swiginit(self, _array.new_doubleArray(*args))

        if len(args) == 1 and isinstance(args[0], list):
            self.MakeDataOwner()



    __swig_destroy__ = _array.delete_doubleArray

    def GetData(self, *args):
        r"""
        GetData(doubleArray self) -> double
        GetData(doubleArray self) -> double const *
        """
        return _array.doubleArray_GetData(self, *args)

    def GetMemory(self, *args):
        r"""
        GetMemory(doubleArray self) -> mfem::Memory< double >
        GetMemory(doubleArray self) -> mfem::Memory< double > const &
        """
        return _array.doubleArray_GetMemory(self, *args)

    def UseDevice(self):
        r"""UseDevice(doubleArray self) -> bool"""
        return _array.doubleArray_UseDevice(self)

    def OwnsData(self):
        r"""OwnsData(doubleArray self) -> bool"""
        return _array.doubleArray_OwnsData(self)

    def StealData(self, p):
        r"""StealData(doubleArray self, double ** p)"""
        return _array.doubleArray_StealData(self, p)

    def LoseData(self):
        r"""LoseData(doubleArray self)"""
        return _array.doubleArray_LoseData(self)

    def MakeDataOwner(self):
        r"""MakeDataOwner(doubleArray self)"""
        return _array.doubleArray_MakeDataOwner(self)

    def Size(self):
        r"""Size(doubleArray self) -> int"""
        return _array.doubleArray_Size(self)

    def SetSize(self, *args):
        r"""
        SetSize(doubleArray self, int nsize)
        SetSize(doubleArray self, int nsize, double const & initval)
        SetSize(doubleArray self, int nsize, mfem::MemoryType mt)
        """
        return _array.doubleArray_SetSize(self, *args)

    def Capacity(self):
        r"""Capacity(doubleArray self) -> int"""
        return _array.doubleArray_Capacity(self)

    def Reserve(self, capacity):
        r"""Reserve(doubleArray self, int capacity)"""
        return _array.doubleArray_Reserve(self, capacity)

    def Append(self, *args):
        r"""
        Append(doubleArray self, double const & el) -> int
        Append(doubleArray self, double const * els, int nels) -> int
        Append(doubleArray self, doubleArray els) -> int
        """
        return _array.doubleArray_Append(self, *args)

    def Prepend(self, el):
        r"""Prepend(doubleArray self, double const & el) -> int"""
        return _array.doubleArray_Prepend(self, el)

    def Last(self, *args):
        r"""
        Last(doubleArray self) -> double
        Last(doubleArray self) -> double const &
        """
        return _array.doubleArray_Last(self, *args)

    def Union(self, el):
        r"""Union(doubleArray self, double const & el) -> int"""
        return _array.doubleArray_Union(self, el)

    def Find(self, el):
        r"""Find(doubleArray self, double const & el) -> int"""
        return _array.doubleArray_Find(self, el)

    def FindSorted(self, el):
        r"""FindSorted(doubleArray self, double const & el) -> int"""
        return _array.doubleArray_FindSorted(self, el)

    def DeleteLast(self):
        r"""DeleteLast(doubleArray self)"""
        return _array.doubleArray_DeleteLast(self)

    def DeleteFirst(self, el):
        r"""DeleteFirst(doubleArray self, double const & el)"""
        return _array.doubleArray_DeleteFirst(self, el)

    def DeleteAll(self):
        r"""DeleteAll(doubleArray self)"""
        return _array.doubleArray_DeleteAll(self)

    def Copy(self, copy):
        r"""Copy(doubleArray self, doubleArray copy)"""
        return _array.doubleArray_Copy(self, copy)

    def MakeRef(self, *args):
        r"""
        MakeRef(doubleArray self, double * arg2, int arg3)
        MakeRef(doubleArray self, doubleArray master)
        """
        return _array.doubleArray_MakeRef(self, *args)

    def GetSubArray(self, offset, sa_size, sa):
        r"""GetSubArray(doubleArray self, int offset, int sa_size, doubleArray sa)"""
        return _array.doubleArray_GetSubArray(self, offset, sa_size, sa)

    def Load(self, *args):
        r"""
        Load(doubleArray self, std::istream & _in, int fmt=0)
        Load(doubleArray self, int new_size, std::istream & _in)
        """
        return _array.doubleArray_Load(self, *args)

    def Max(self):
        r"""Max(doubleArray self) -> double"""
        return _array.doubleArray_Max(self)

    def Min(self):
        r"""Min(doubleArray self) -> double"""
        return _array.doubleArray_Min(self)

    def Sort(self):
        r"""Sort(doubleArray self)"""
        return _array.doubleArray_Sort(self)

    def Unique(self):
        r"""Unique(doubleArray self)"""
        return _array.doubleArray_Unique(self)

    def IsSorted(self):
        r"""IsSorted(doubleArray self) -> int"""
        return _array.doubleArray_IsSorted(self)

    def PartialSum(self):
        r"""PartialSum(doubleArray self)"""
        return _array.doubleArray_PartialSum(self)

    def Sum(self):
        r"""Sum(doubleArray self) -> double"""
        return _array.doubleArray_Sum(self)

    def begin(self):
        r"""begin(doubleArray self) -> double *"""
        return _array.doubleArray_begin(self)

    def end(self):
        r"""end(doubleArray self) -> double *"""
        return _array.doubleArray_end(self)

    def MemoryUsage(self):
        r"""MemoryUsage(doubleArray self) -> long"""
        return _array.doubleArray_MemoryUsage(self)

    def Read(self, on_dev=True):
        r"""Read(doubleArray self, bool on_dev=True) -> double const"""
        return _array.doubleArray_Read(self, on_dev)

    def HostRead(self):
        r"""HostRead(doubleArray self) -> double const *"""
        return _array.doubleArray_HostRead(self)

    def Write(self, on_dev=True):
        r"""Write(doubleArray self, bool on_dev=True) -> double"""
        return _array.doubleArray_Write(self, on_dev)

    def HostWrite(self):
        r"""HostWrite(doubleArray self) -> double *"""
        return _array.doubleArray_HostWrite(self)

    def ReadWrite(self, on_dev=True):
        r"""ReadWrite(doubleArray self, bool on_dev=True) -> double"""
        return _array.doubleArray_ReadWrite(self, on_dev)

    def HostReadWrite(self):
        r"""HostReadWrite(doubleArray self) -> double *"""
        return _array.doubleArray_HostReadWrite(self)

    def __setitem__(self, i, v):
        r"""__setitem__(doubleArray self, int i, double const v)"""
        return _array.doubleArray___setitem__(self, i, v)

    def __getitem__(self, i):
        r"""__getitem__(doubleArray self, int const i) -> double const &"""
        return _array.doubleArray___getitem__(self, i)

    def Assign(self, *args):
        r"""
        Assign(doubleArray self, double const * arg2)
        Assign(doubleArray self, double const & a)
        """
        return _array.doubleArray_Assign(self, *args)

    def ToList(self):
        return [self[i] for i in range(self.Size())]



    def Print(self, *args):
        r"""
        Print(doubleArray self, std::ostream & out=mfem::out, int width=4)
        Print(doubleArray self, char const * file, int precision=8)
        """
        return _array.doubleArray_Print(self, *args)

    def Save(self, *args):
        r"""
        Save(doubleArray self, std::ostream & out, int fmt=0)
        Save(doubleArray self, char const * file, int precision=8)
        Save(doubleArray self)
        """
        return _array.doubleArray_Save(self, *args)

# Register doubleArray in _array:
_array.doubleArray_swigregister(doubleArray)


def doubleSwap(*args):
    r"""
    doubleSwap(doubleArray arg1, doubleArray arg2)
    doubleSwap(mfem::Array2D< double > & arg1, mfem::Array2D< double > & arg2)
    doubleSwap(double & a, double & b)
    """
    return _array.doubleSwap(*args)

def intSwap(*args):
    r"""
    intSwap(intArray arg1, intArray arg2)
    intSwap(mfem::Array2D< int > & arg1, mfem::Array2D< int > & arg2)
    intSwap(int & a, int & b)
    """
    return _array.intSwap(*args)


