#   This code calculates a probability distribution using Bayes filter after some sensings and movings 
#   of localization problem.    
#   Some important variables:
#   - env:
#       2D list, each entry either 'R' (for red cell) or 'G' (for green cell).
#   - sensor_right:
#       The probability that any given measurement of env is correct.
#   - motion_success:
#       The probability that any movement is successful.
#   - measurements:
#       List of measurements taken by the robot, each entry either 'R' or 'G'
#   - motions:
#       List of actions taken by the robot, each entry of the form [dy,dx], i.e.,
#       [0,0] - stay
#       [0,1] - right
#       [0,-1] - left
#       [1,0] - down
#       [-1,0] - up
#        
#   Also assume that at each step, the robot:
#   1) first makes a movement,
#   2) then takes a measurement.
#

import numpy as np

def initialize(env):
    # create uniform probability distribution
    p = np.ones((len(env),len(env[0])))
    p = p / len(env)  / len(env[0])
    return p 


def sense(p, measurement, sensor_right, env):
    next_p = np.zeros((len(p), len(p[0])))
    sensor_worng = 1 - sensor_right
    for col in range(len(p)):
        for row in range(len(p[0])):
            measure_is_same = (measurement == env[col][row])
            next_p[col][row] = p[col][row] * (measure_is_same * sensor_right + (1 - measure_is_same) * sensor_worng)

    norm_ratio = np.sum(next_p)
    for col in range(len(p)):
        for row in range(len(p[0])):
            next_p[col][row] = next_p[col][row] / norm_ratio

    return next_p

def move(p, motion, move_success):
    next_p = np.zeros((len(p), len(p[0])))
    move_unsucc = 1 - move_success

    for col in range(len(p)):
        for row in range(len(p[0])):
            next_p[col][row] =  p[(col - motion[0]) % len(p)][(row - motion[1]) % len(p[0])] * move_success + p[col][row] * move_unsucc
    
    return next_p

def measure_except_check(measurements):
    for color in measurements:
        if color != 'G' and color != 'R':
            raise ValueError, "Error in value of measurement vector"

def motion_except_check(motions):
    for motion in motions:
        if motion != [0,0] and motion != [0,1] and motion != [0,-1] and motion != [1,0] and motion != [-1,0]:
            raise ValueError, "Error in value of motion vector"

def measure_motion_size_check(measurements, motions):
    if len(measurements) != len(motions):
        raise ValueError, "Error in size of measurement/motion vector"
    
    
def show(p):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x),r)) + ']' for r in p]
    print '[' + ',\n '.join(rows) + ']'


def main():

    env = [['R','G','G','R','R'],
           ['R','R','G','R','R'],
           ['R','R','G','G','R'],
           ['R','R','R','R','R']]
    
    sensor_right = 0.7
    move_success = 0.8

    measurements = ['G', 'G', 'G', 'G', 'G']
    motions = [[0,0], [0,1], [1,0], [1,0], [0,1]]

    measure_except_check(measurements)
    motion_except_check(motions)
    measure_motion_size_check(measurements, motions)

    p = initialize(env)
    
    for i in range(len(measurements)):
        p = move(p, motions[i], move_success)
        p = sense(p, measurements[i], sensor_right, env)

    show(p)

if __name__ == "__main__":
    main()
