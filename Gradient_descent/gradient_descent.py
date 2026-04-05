import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 5000
    n = len(x)
    learning_rate = 0.08
    costs = []
    iters = []

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum((y-y_predicted)**2)
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # save cost every 100 iterations for plotting
        if i % 100 == 0:
            costs.append(cost)
            iters.append(i)
            print(f"m: {m_curr:.4f}, b: {b_curr:.4f}, cost: {cost:.4f}, iteration: {i}")

    # Plot cost vs iteration
    plt.plot(iters, costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.show()
   
    # Final regression line plot
    y_pred_final = m_curr * x + b_curr
    plt.scatter(x, y, color="red", label="Data points")
    plt.plot(x, y_pred_final, color="blue", label=f"Regression line: y={m_curr:.2f}x+{b_curr:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x, y)