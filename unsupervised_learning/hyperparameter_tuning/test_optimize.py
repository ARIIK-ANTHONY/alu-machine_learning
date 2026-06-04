#!/usr/bin/env python3

BO = __import__('5-bayes_opt').BayesianOptimization
import numpy as np

def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)
    
    # Create Bayesian Optimization with proper parameters
    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, 
            l=0.6, sigma_f=2, xsi=0.01, minimize=True)
    
    print("Initial samples:")
    for i in range(len(bo.gp.X)):
        print(f"  X={bo.gp.X[i][0]:.8f}, Y={bo.gp.Y[i][0]:.8f}")
    print()
    
    X_opt, Y_opt = bo.optimize(50)
    
    print(f'Optimal X: [{X_opt[0]:.8f}]')
    print(f'Optimal Y: [{Y_opt[0]:.8f}]')
    print(f'All sample inputs:')
    for sample in bo.gp.X:
        print(f' [{sample[0]:.8f}]')
    print(f'Total samples: {len(bo.gp.X)}')
    
    # Check if we found the global optimum
    expected_X = 0.8975979
    if abs(X_opt[0] - expected_X) < 1e-5:
        print("\n✓ SUCCESS: Found the correct global optimum!")
    else:
        print(f"\n✗ Found local optimum at {X_opt[0]:.8f}")
        print(f"  Expected global optimum at {expected_X}")
