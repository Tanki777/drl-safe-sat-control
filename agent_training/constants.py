"""
Contains all constants.

Author: Orfeas Koulamas - 2026
Modified by: Cemal Yilmaz - 2026
"""

#This file is imported by SafetyFilter.py, constraintE.py, and constraintQ.py
import numpy as np

### PHYSICAL PARAMETERS
J_yang = np.array([[60.0, 5.0, 1.0],
          [5.0, 50.0, 2.0],
          [1.0, 2.0, 70.0]],
          np.float32)

u_max_yang = 2.0 # Nm

J_b = np.diag([0.1672, 0.1259, 0.06121])  # Body's moment of inertia (no wheels) [kg·m²] (diagonal 3x3)

J_w = 0.00001722  # One wheel's moment of inertia [kg·m²]

# Using only 3 reaction wheels (deactivating the 4th wheel for simplified training)
A=np.array([[0.0,  0.0,    0.8165],  # Wheel positions matrix (in body frame) - only first 3 wheels
            [0.0, -0.9428, 0.4714],    # each column is one a_i, like in 2023 paper 
            [-1.0, 0.3333, 0.3333]])   # 4th wheel is permanently deactivated  

J_tot = J_b + J_w * A @ A.transpose()  # see 3 lines above eq. (2a) in 2023 paper 
# A @ A.T is a 3x3 matrix equal to the sum_i{(a_i)(a_i)T} (the sum of a_i's outer products)

#u_max = 7e-4  # Max wheel torque [N·m]
u_max = 0.001
w_max = 628.3 # Max wheel speed [rad/s] (for M1omega calculation)
xi_max = 1e-5 # Max disturbance torque [N·m]
e_max = 5.092e-5 # Max allowed kinetic energy [kg·m²/s²]

dt = 0.1  # Control timestep (how often do we update) [seconds]


### POINTING CONSTRAINT PARAMETERS
r_F = np.array([1, 0, 0]) # Boresight vector in body frame

# CBF parameters
mu = 0.00167          
M2plus  =  1.64e-4
M2minus = -1.64e-4 
M3plus  =  6.2e-3  
M3minus = -6.2e-3

### COMPUTED PARAMETERS (Don't edit these, they're auto-calculated)

Jw_matrix = np.diag([J_w, J_w, J_w])  # turns scalar J_w value into 3x3 diagonal matrix (only 3 wheels)
    
# Build Z matrix
top = np.hstack([J_tot, A @ Jw_matrix])             # top dimension   : (3x3) + (3x3)(3x3) = (3x3) + (3x3) = (3x6)
bottom = np.hstack([Jw_matrix @ A.T, Jw_matrix])    # bottom dimension: (3x3)(3x3) + (3x3) = (3x3) + (3x3) = (3x6)
M = np.vstack([top, bottom])                        # total dimension : (6x6) matrix 
Z = np.linalg.inv(M)                                # Z is also (6x6) as the inverse of M
    
Je = np.linalg.inv(Z[0:3, 0:3]) # used for constraintE

#calculate δ1 from 2023 paper's eq.(24)
tau = np.linspace(0.0, dt, 10001) # discretize the range [0,dt] (dt is the same as [0,T] in (24)), (10000 divisions)
expr1 = 0.5*(mu + M2plus - M2minus)*(dt - tau)**2 - (1/6)*M3minus*(dt - tau)**3 # 1st expression array
expr2 = 0.5*(mu + M2plus - M2minus)*tau**2 + (1/6)*M3plus*tau**3 # 2nd expression array
delta1 = np.max(np.minimum(expr1, expr2)) # finds the single max value of the min(expr1,expr2) array

delta_2 = delta1 # as seen in one paragraph below the paper's eq.(25):
Delta_2 = delta1 # "It follows from Theorem 1 that one valid combination is Δ2 = δ2 = δ1"
    
# M1omega and M2omega are used in ConstraintE
M1omega = 2 * w_max * u_max * np.sqrt(Je[0,0] / Je[2,2]) # Compute M1omega from wheel parameters
M2omega = 1.95e-5 # Fixed value from paper, and same in their code


### EXPORT AS DICTIONARY
def get_constants(): # Returns a dictionary with all constants needed by the safety filter.
    
    constants = {
        # Computed dynamics
        'Z': Z,
        'Je': Je,
        
        # Energy constraint
        'e_max': e_max,
        'M1omega': M1omega,
        'M2omega': M2omega,
        
        # Pointing constraint
        'r_F' : r_F,
        'mu': mu,
        'M2plus': M2plus,
        'M3plus': M3plus,
        'delta_2': delta_2,
        'Delta_2': Delta_2,
        
        # Control parameters
        'dt': dt,
        'u_max': u_max,
        
        # Physical parameters (for reference)
        'J_tot': J_tot,
        'A': A,
        'J_w': Jw_matrix,
        "J_yang": J_yang,
        "u_max_yang": u_max_yang,
        "J_b": J_b,
        "J_w_scalar": J_w
    }
    
    return constants
