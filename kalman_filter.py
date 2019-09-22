#   This code implements the Kalman filter of moving vehicle on a 2D axixs 
#   with the dynamics:   
#
#   X(k+1) = AX(k)
#   Y(k) = CX(k) + V(k)
# 
#   where X(k) = [x1 x2 v1 v2]^T is the state, Y(k) the measurement, and V(k) the measurement noise.
#   Note: No control input nor process noise. 
#
#   Ref: Udacity course, Artificial Intelligence for Robotic
#   https://classroom.udacity.com/courses/cs373/
#

import numpy as np

def predict(X, P, A):
    X = np.dot(A, X)
    P = np.dot(np.dot(A, P), A.T)

    return X, P

def estimate(X, P, A, Y, C, R):
    S = np.dot(np.dot(C, P), C.T) + R 
    K = np.dot(np.dot(P, C.T), np.linalg.inv(S))
    X = X + np.dot(K, Y - np.dot(C, X))
    P = np.dot(np.eye(len(X)) - np.dot(K, C), P)

    return X, P

def show(M):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x),r)) + ']' for r in M]
    print '[' + ',\n '.join(rows) + ']'


def main():

    Yt = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
    x0 = [4., 12.]

    # measurements = [[1., 4.], [6., 0.], [11., -4.], [16., -8.]]
    # X0 = [-4., 8.]

    # measurements = [[1., 17.], [1., 15.], [1., 13.], [1., 11.]]
    # X0 = [1., 19.]

    dt = 0.1

    X = np.array([[x0[0]], [x0[1]], [0.], [0.]])
    P = np.array([[0., 0., 0., 0.], 
                  [0., 0., 0., 0.], 
                  [0., 0., 100., 0.], 
                  [0., 0., 0., 100.]])
    A = np.array([[1., 0., dt, 0.], 
                  [0., 1., 0., dt], 
                  [0., 0., 1., 0.], 
                  [0., 0., 0., 1.]])
    C = np.array([[1., 0., 0, 0], 
                  [0., 1., 0., 0.]])
    R = np.array([[0.1, 0], 
                  [0, 0.1]])
    
    for i in range(len(Yt)):
        [X, P] = predict(X, P, A)
        [X, P] = estimate(X, P, A, Yt[i], C, R)

    show(P)



if __name__ == "__main__":
    main()
