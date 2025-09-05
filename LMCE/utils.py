import cvxpy as cp
import numpy as np


def min_max(to_scale: np.ndarray,
            mins: np.ndarray=np.array([]),
            maxs: np.ndarray=np.array([])):

    vec_d = to_scale.shape[1]
    if not len(mins):
        maxs = np.array([max(to_scale[:, i]) for i in range(vec_d)])
        mins = np.array([min(to_scale[:, i]) for i in range(vec_d)])

    scaled = np.zeros(to_scale.shape)
    for i, _ in enumerate(to_scale):
        for j, _ in enumerate(to_scale[0]):
            scaled[i][j] = (((to_scale[i][j]-mins[j])/(maxs[j]-mins[j]))-.5)*2

    return scaled, mins, maxs


class SplineFitter:
    def __init__(self):
        pass

    def fit(self, t, y, num_segments = 5, degree = 4):
        a = 0.000001 #50000 # weight for the least square error 
        l = 0.0 #0.1 #0.0000001 # weight for the regularization
        coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
        T = t[-1] - t[0]
        T_segment = T / num_segments
        cost = 0
        constraints = []

        j = 0
        for i in range(num_segments):
            start_id = j
            while j < y.shape[0]:
                t_normalized = (t[j] - i*T_segment)/T_segment
                if t_normalized > 1.0:
                    break
                y_poly = sum([coeffs[i][d]*t_normalized**d for d in range(degree+1)])
                cost += a*cp.sum_squares(y_poly - y[j])
                j = j + 1
            cost += l * cp.sum_squares(coeffs[i])

            if i < num_segments - 1:
                x = 1.0 #data_points[end_id - 1][0]
                # Calculate the derivatives at the boundary
                boundary_value    = sum(coeffs[i][d]*x**d for d in range(degree + 1))
                dboundary_value   = sum(d * coeffs[i][d] * x**(d - 1) for d in range(1, degree + 1))
                ddboundary_value  = sum(d * (d - 1) * coeffs[i][d] * x**(d - 2) for d in range(2, degree + 1))
                dddboundary_value  = sum(d * (d - 2) * (d - 1) * coeffs[i][d] * x**(d - 3) for d in range(3, degree + 1))
                x = 0.0
                boundary_value_next   = sum(coeffs[i + 1][d] * x**d for d in range(degree + 1))
                dboundary_value_next  = sum(d * coeffs[i + 1][d] * x**(d - 1) for d in range(1, degree + 1))
                ddboundary_value_next = sum(d * (d - 1) * coeffs[i + 1][d] * x**(d - 2) for d in range(2, degree + 1))
                # dddboundary_value_next = sum(d * (d - 2) * (d - 1) * coeffs[i + 1][d] * x**(d - 3) for d in range(3, degree + 1))
            # add constraints
                constraints.append(boundary_value == boundary_value_next)
                constraints.append(dboundary_value == dboundary_value_next)
                constraints.append(ddboundary_value == ddboundary_value_next)
                # constraints.append(dddboundary_value == dddboundary_value_next)
            
                if degree > 3:
                    x = 1.0
                    dddboundary_value  = sum(d * (d - 1) * (d - 2) * coeffs[i][d] * x**(d - 3) for d in range(3, degree + 1))
                    x = 0.0
                    dddboundary_value_next = sum(d * (d - 1) * (d - 2) * coeffs[i + 1][d] * x**(d - 3) for d in range(3, degree + 1))
                    constraints.append(dddboundary_value == dddboundary_value_next)

                
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP)

        self.coeffs = coeffs
        self.T_segment = T_segment
        self.num_segments = num_segments
        self.degree = degree

    def eval(self, t):
        result = []
        for t_global in t:
            i = (int)(t_global / self.T_segment)
            t_normalized = (t_global - i*self.T_segment)/self.T_segment
            if t_normalized == 0.0:
                i = i - 1
                t_normalized = 1.0
            y_poly = sum([self.coeffs[i][d].value*t_normalized**d for d in range(self.degree+1)])
            result.append(y_poly)
        return np.array(result)
    
    def evald(self, t):
        result = []
        for t_global in t:
            i = (int)(t_global / self.T_segment)
            t_normalized = (t_global - i*self.T_segment)/self.T_segment
            if t_normalized == 0.0:
                i = i - 1
                t_normalized = 1.0
            y_poly = sum([d * self.coeffs[i][d].value*t_normalized**(d-1)/self.T_segment for d in range(1, self.degree+1)])
            result.append(y_poly)
        return np.array(result)
    
    def evaldd(self, t):
        result = []
        for t_global in t:
            i = (int)(t_global / self.T_segment)
            t_normalized = (t_global - i*self.T_segment)/self.T_segment
            if t_normalized == 0.0:
                i = i - 1
                t_normalized = 1.0
            y_poly = sum([d * (d-1) * self.coeffs[i][d].value*t_normalized**(d-2)/(self.T_segment**2) for d in range(2, self.degree+1)])
            result.append(y_poly)
        return np.array(result)
