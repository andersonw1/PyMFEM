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
    from . import _device
else:
    import _device

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


import mfem._par.mem_manager
class Backend(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    CPU = _device.Backend_CPU
    OMP = _device.Backend_OMP
    CUDA = _device.Backend_CUDA
    RAJA_CPU = _device.Backend_RAJA_CPU
    RAJA_OMP = _device.Backend_RAJA_OMP
    RAJA_CUDA = _device.Backend_RAJA_CUDA
    OCCA_CPU = _device.Backend_OCCA_CPU
    OCCA_OMP = _device.Backend_OCCA_OMP
    OCCA_CUDA = _device.Backend_OCCA_CUDA
    NUM_BACKENDS = _device.Backend_NUM_BACKENDS
    CPU_MASK = _device.Backend_CPU_MASK
    CUDA_MASK = _device.Backend_CUDA_MASK
    OMP_MASK = _device.Backend_OMP_MASK
    DEVICE_MASK = _device.Backend_DEVICE_MASK
    RAJA_MASK = _device.Backend_RAJA_MASK
    OCCA_MASK = _device.Backend_OCCA_MASK

    def __init__(self):
        _device.Backend_swiginit(self, _device.new_Backend())
    __swig_destroy__ = _device.delete_Backend

# Register Backend in _device:
_device.Backend_swigregister(Backend)

class Device(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _device.Device_swiginit(self, _device.new_Device(*args))
    __swig_destroy__ = _device.delete_Device

    def Configure(self, device, dev=0):
        return _device.Device_Configure(self, device, dev)

    def Print(self, *args):
        return _device.Device_Print(self, *args)

    @staticmethod
    def IsConfigured():
        return _device.Device_IsConfigured()

    @staticmethod
    def IsAvailable():
        return _device.Device_IsAvailable()

    @staticmethod
    def IsEnabled():
        return _device.Device_IsEnabled()

    @staticmethod
    def IsDisabled():
        return _device.Device_IsDisabled()

    @staticmethod
    def Allows(b_mask):
        return _device.Device_Allows(b_mask)

    @staticmethod
    def GetMemoryType():
        return _device.Device_GetMemoryType()

    @staticmethod
    def GetMemoryClass():
        return _device.Device_GetMemoryClass()

# Register Device in _device:
_device.Device_swigregister(Device)

def Device_IsConfigured():
    return _device.Device_IsConfigured()

def Device_IsAvailable():
    return _device.Device_IsAvailable()

def Device_IsEnabled():
    return _device.Device_IsEnabled()

def Device_IsDisabled():
    return _device.Device_IsDisabled()

def Device_Allows(b_mask):
    return _device.Device_Allows(b_mask)

def Device_GetMemoryType():
    return _device.Device_GetMemoryType()

def Device_GetMemoryClass():
    return _device.Device_GetMemoryClass()



