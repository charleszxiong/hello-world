

import numpy as np
from scipy import optimize


def adam(fun, x0, args=(), jac=None, tol=None,
         eps=1e-8, maxiter=np.inf,
         eta=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
         **kwargs
        ):
    '''Minimize objective function using the Adam algorithm
    
    Parameters
    ------------
    fun: function
        Scalar objective function whose input is an ndarray of shape (n, ).
    
    x0: ndarray of shape (n, )
        Initial guess.
        
    args: tuple
        Extra arguments to pass to objective function and Jacobian.
    
    jac: function
        Jacobian function, whose input and output are ndarrays of shape (n, ).
        If None, will be calculated using finite difference.
        
    tol: float
        Tolerance for termination. If None, default value 1e-10 is used.
    
    eps: float
        Step size used in finite difference approximation of jac.

    maxiter: integer
        Maximum number of iterations.

    eta: float
        Learning rate.

    beta_1: float
        Decay rate of first moment. 0 <= beta_1 < 1.

    beta_2: float
        Decay rate of second moment. 0 <= beta_2 < 1.

    epsilon: float
        Smoothing parameter to avoid division by zero. 
    
    kwargs: keywords
        Receptacle of additional keywords that are to be ignored.
    '''
    # If tolerance for termination set to None, reset it
    if tol is None:
        tol = 1e-10
        
    # If Jacobian not specified, use finite difference approximation
    if jac is None:
        fx = lambda x: fun(x, *args)
        jac = lambda x: optimize.approx_fprime(x, fx, eps)
    
    # Get dimension of input
    n = len(x0)
    
    # Initialize
    x_t = np.array(x0)
    m_t = np.zeros(n)
    v_t = np.zeros(n)
    t = 0    # iteration
    nfev = 0 # number of function evaluations
    njev = 0 # number of Jacobian evaluations
    
    while t < maxiter:
        # Update iteration number
        t = t + 1
        # Current gradient
        g_t = jac(x_t)
        njev = njev + 1
        # Moments of accumulated gradient
        m_t = beta_1 * m_t + (1 - beta_1) * g_t
        v_t = beta_2 * v_t + (1 - beta_2) * g_t ** 2
        # Bias-corrected moments
        m_t_hat = m_t / (1 - beta_1 ** t)
        v_t_hat = v_t / (1 - beta_2 ** t)
        # Coordinate update
        delta_x_t = - eta * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
        x_t = x_t + delta_x_t
        if np.linalg.norm(delta_x_t) < tol:
            break
    
    # Summarize results
    success = (t < maxiter)
    if success:
        message = 'Optimizer converged.'
    else:
        message = 'Maximum number of iterations reached.'
    
    # Return result
    return optimize.OptimizeResult(x=x_t,
                                   success=success,
                                   message=message,
                                   fun=fun(x_t),
                                   jac=jac(x_t),
                                   nfev=nfev,
                                   njev=njev,
                                   nit=t
                                  )