/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * This file is not intended to be easily readable and contains a number of
 * coding conventions designed to improve portability and efficiency. Do not make
 * changes to this file unless you know what you are doing--modify the SWIG
 * interface file instead.
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_coefficient_WRAP_H_
#define SWIG_coefficient_WRAP_H_

#include <map>
#include <string>


class SwigDirector_PyCoefficientBase : public mfem::PyCoefficientBase, public Swig::Director {

public:
    SwigDirector_PyCoefficientBase(PyObject *self, int tdep);
    virtual double Eval(mfem::ElementTransformation &T, mfem::IntegrationPoint const &ip);
    virtual ~SwigDirector_PyCoefficientBase();
    virtual double _EvalPy(mfem::Vector &arg0);
    virtual double _EvalPyT(mfem::Vector &arg0, double arg1);

/* Internal director utilities */
public:
    bool swig_get_inner(const char *swig_protected_method_name) const {
      std::map<std::string, bool>::const_iterator iv = swig_inner.find(swig_protected_method_name);
      return (iv != swig_inner.end() ? iv->second : false);
    }
    void swig_set_inner(const char *swig_protected_method_name, bool swig_val) const {
      swig_inner[swig_protected_method_name] = swig_val;
    }
private:
    mutable std::map<std::string, bool> swig_inner;

#if defined(SWIG_PYTHON_DIRECTOR_VTABLE)
/* VTable implementation */
    PyObject *swig_get_method(size_t method_index, const char *method_name) const {
      PyObject *method = vtable[method_index];
      if (!method) {
        swig::SwigVar_PyObject name = SWIG_Python_str_FromChar(method_name);
        method = PyObject_GetAttr(swig_get_self(), name);
        if (!method) {
          std::string msg = "Method in class PyCoefficientBase doesn't exist, undefined ";
          msg += method_name;
          Swig::DirectorMethodException::raise(msg.c_str());
        }
        vtable[method_index] = method;
      }
      return method;
    }
private:
    mutable swig::SwigVar_PyObject vtable[3];
#endif

};


class SwigDirector_VectorPyCoefficientBase : public mfem::VectorPyCoefficientBase, public Swig::Director {

public:
    SwigDirector_VectorPyCoefficientBase(PyObject *self, int dim, int tdep, mfem::Coefficient *q = NULL);
    virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T, mfem::IntegrationPoint const &ip);
    virtual void Eval(mfem::DenseMatrix &M, mfem::ElementTransformation &T, mfem::IntegrationRule const &ir);
    virtual ~SwigDirector_VectorPyCoefficientBase();
    virtual void _EvalPy(mfem::Vector &arg0, mfem::Vector &arg1);
    virtual void _EvalPyT(mfem::Vector &arg0, double arg1, mfem::Vector &arg2);

/* Internal director utilities */
public:
    bool swig_get_inner(const char *swig_protected_method_name) const {
      std::map<std::string, bool>::const_iterator iv = swig_inner.find(swig_protected_method_name);
      return (iv != swig_inner.end() ? iv->second : false);
    }
    void swig_set_inner(const char *swig_protected_method_name, bool swig_val) const {
      swig_inner[swig_protected_method_name] = swig_val;
    }
private:
    mutable std::map<std::string, bool> swig_inner;

#if defined(SWIG_PYTHON_DIRECTOR_VTABLE)
/* VTable implementation */
    PyObject *swig_get_method(size_t method_index, const char *method_name) const {
      PyObject *method = vtable[method_index];
      if (!method) {
        swig::SwigVar_PyObject name = SWIG_Python_str_FromChar(method_name);
        method = PyObject_GetAttr(swig_get_self(), name);
        if (!method) {
          std::string msg = "Method in class VectorPyCoefficientBase doesn't exist, undefined ";
          msg += method_name;
          Swig::DirectorMethodException::raise(msg.c_str());
        }
        vtable[method_index] = method;
      }
      return method;
    }
private:
    mutable swig::SwigVar_PyObject vtable[4];
#endif

};


class SwigDirector_MatrixPyCoefficientBase : public mfem::MatrixPyCoefficientBase, public Swig::Director {

public:
    SwigDirector_MatrixPyCoefficientBase(PyObject *self, int dim, int tdep);
    virtual void Eval(mfem::DenseMatrix &K, mfem::ElementTransformation &T, mfem::IntegrationPoint const &ip);
    virtual ~SwigDirector_MatrixPyCoefficientBase();
    virtual void _EvalPy(mfem::Vector &arg0, mfem::DenseMatrix &arg1);
    virtual void _EvalPyT(mfem::Vector &arg0, double arg1, mfem::DenseMatrix &arg2);

/* Internal director utilities */
public:
    bool swig_get_inner(const char *swig_protected_method_name) const {
      std::map<std::string, bool>::const_iterator iv = swig_inner.find(swig_protected_method_name);
      return (iv != swig_inner.end() ? iv->second : false);
    }
    void swig_set_inner(const char *swig_protected_method_name, bool swig_val) const {
      swig_inner[swig_protected_method_name] = swig_val;
    }
private:
    mutable std::map<std::string, bool> swig_inner;

#if defined(SWIG_PYTHON_DIRECTOR_VTABLE)
/* VTable implementation */
    PyObject *swig_get_method(size_t method_index, const char *method_name) const {
      PyObject *method = vtable[method_index];
      if (!method) {
        swig::SwigVar_PyObject name = SWIG_Python_str_FromChar(method_name);
        method = PyObject_GetAttr(swig_get_self(), name);
        if (!method) {
          std::string msg = "Method in class MatrixPyCoefficientBase doesn't exist, undefined ";
          msg += method_name;
          Swig::DirectorMethodException::raise(msg.c_str());
        }
        vtable[method_index] = method;
      }
      return method;
    }
private:
    mutable swig::SwigVar_PyObject vtable[3];
#endif

};


#endif
