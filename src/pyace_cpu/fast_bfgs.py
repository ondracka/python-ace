"""Equivalent O(N^2) BFGS update, with an explicit process-local enable hook.

SciPy optimize.py module by Travis E. Oliphant.
You may copy and use this module as you see fit with no guarantee implied
provided you keep this notice in all copies.
"""
import inspect
import numpy as np
from scipy.linalg.blas import dger


def rank2_update(hessian, step, gradient_change, rho):
    """Expanded BFGS formula using BLAS rank-one updates, in-place for F arrays.

    Retain both H*y and H.T*y so the expansion is also valid if rounding has
    introduced a small asymmetry. No Hessian approximation or history truncation.
    """
    hessian = np.asfortranarray(hessian, dtype=np.float64)
    hy = hessian @ gradient_change
    y_h = hessian.T @ gradient_change
    scale = rho + rho ** 2 * np.dot(gradient_change, hy)
    hessian = dger(-rho, step, y_h, a=hessian, overwrite_a=True)
    hessian = dger(-rho, hy, step, a=hessian, overwrite_a=True)
    return dger(scale, step, step, a=hessian, overwrite_a=True)


def enable():
    """Keep SciPy's line search/stopping/callbacks, replace only matrix algebra.

    No site-packages files are written. Exact source guards fail closed if an
    incompatible SciPy implementation is installed. Retain the ordinary BFGS
    algorithm and all coefficients, not a limited-memory approximation.
    """
    from scipy.optimize import _optimize, _minimize
    original = _optimize._minimize_bfgs
    if getattr(original, '_wbc_rank2', False):
        return
    source = inspect.getsource(original)
    replacements = [
        ('    I = np.eye(N, dtype=int)\n'
         '    Hk = I if hess_inv0 is None else hess_inv0',
         '    Hk = (np.eye(N, dtype=float, order="F") if hess_inv0 is None\n'
         '          else np.array(hess_inv0, dtype=float, order="F", copy=True))'),
        ('        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok\n'
         '        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok\n'
         '        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *\n'
         '                                                 sk[np.newaxis, :])',
         '        Hk = _wbc_rank2_update(Hk, sk, yk, rhok)')]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('SciPy BFGS source differs from the verified implementation')
        source = source.replace(old, new)
    namespace = dict(original.__globals__, _wbc_rank2_update=rank2_update)
    exec(compile(source, '<WBC rank-two SciPy BFGS>', 'exec'), namespace)
    optimized = namespace['_minimize_bfgs']
    optimized._wbc_rank2 = True
    _optimize._minimize_bfgs = optimized
    _minimize._minimize_bfgs = optimized
