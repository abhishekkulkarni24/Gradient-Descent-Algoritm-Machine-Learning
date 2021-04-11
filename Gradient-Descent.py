from numpy import *

def compute_error(b, m, pts):
    totalError = 0
    for i in range(0, len(pts)):
        x = pts[i, 0]
        y = pts[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(pts))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


for i in range(1,7,1):
    points = genfromtxt("data"+str(i)+".csv", delimiter=",")
    learning_rate = 0.001
    initial_b , initial_m , num_iterations = 0 ,0 ,1000

    print ("\nGradient descent for dataset = {0} at b = {1}, m = {2}, error = {3}".format(i,initial_b, initial_m, compute_error(initial_b, initial_m, points)))

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points)))

    print("\n---------------------------------------------------------------------------------------------------------------------------------\n")
