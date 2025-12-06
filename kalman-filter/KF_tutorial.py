
import pandas as pd
import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import expm, inv
from matplotlib import pyplot as plt


df = pd.read_csv("data.csv", delimiter=";").set_index("Step")
df.drop(df.index[-1], axis=0, inplace=True)
data = df[df['Month'].isin([1, 2, 3])]  ## keep only summer months

sT = 3600 * 24
data.index = data.index*3600 / sT  # Change time scale to days
train_data = data.iloc[0:1000]
test_data = data.iloc[1000:]


def build_continuous_matrices(R1, R2, C1, C2, qi, qe, r):
    A = np.array([
        [-(1/(C1*R1) + 1/(C1*R2)),  1/(C1*R2)],
        [ 1/(C2*R2),               -1/(C2*R2)]
    ])

    B = np.array([
        [1/(C1*R1),  0],  # T_ext, q
        [0,           1/C2]
    ])
    
    C = np.array([[0, 1]])  # Output: indoor air temp

    # System and measurement error covariance matrices
    Qc = np.diag([qi**2, qe**2])
    Rc = np.array([[r ** 2]])
    return A, B, C, Qc, Rc

def discretize(time_step: float, Ac: np.ndarray, Bc: np.ndarray, Qc=None, Rc=None):
    F = expm(Ac * time_step)
    G = inv(Ac) @ (F - np.eye(Ac.shape[0])) @ Bc
    if Qc is None or Rc is None:
        return F, G
    else:            
        result = cont2discrete((Ac, np.eye(Ac.shape[0]), np.eye(Ac.shape[0]), Qc), time_step, method='zoh')
        Q = result[3]
        # Measurement error covariance matrix (discrete)
        R = Rc / time_step
        return F, G, Q, R


time = data.index
T_in = np.array(data['Tint'])
T_ext = np.array(data['Text'])
q = np.array(data['Qheat'])
time_step = time[1] - time[0]  # time step in days

T_in_initial =  3
T_w_initial =  2

R1 = 0.001  # K/W
R2 = 0.001    # K/W
C1 = 1e7 / sT    # J/K
C2 =  1e6 / sT   # J/K

qi = 1e-3 * sT**0.5  # process noise for internal temp
qe = 1e-3 * sT**0.5  # process noise for wall temp
# qi = 10 * sT**0.5  # process noise for internal temp
# qe = 10 * sT**0.5  # process noise for wall temp

r = 10e-2              # measurement noise for internal temp

Ac, Bc, H, Qc, Rc = build_continuous_matrices(R1, R2, C1, C2, qi, qe, r)
F, G, Q, R = discretize(time_step, Ac, Bc, Qc, Rc)


## Initialisation
uk = np.vstack([T_ext, q])
xk = np.zeros((F.shape[0], uk.shape[1]))  # initial state
P = np.eye(2)  # initial state
xk[:, 0] = np.array([T_w_initial, T_in_initial])  # initial external temp

# Initialize as lists to preserve dimensions when appending
x_s = []
P_s = []
x_updates = []
P_updates = []
loglike = []

for i in range(1, len(time)):
    uk = np.vstack([T_ext[i], q[i]])
    
    # Prediction step
    x_ = F @ np.array([xk[:, i-1]]).T + G @ uk
    P_ = F @ P @ F.T + Q  # process noise

    # Innovation
    y = np.array([[T_in[i]]])
    epsilun = y - (H @ x_)
    Sigma = H @ P_ @ H.T + R  # innovation covariance
    Kt = P_ @ H.T @ inv(Sigma)  # Kalman gain

    l = 0.5 * (epsilun[0,0] ** 2 / Sigma[0,0] + np.log(Sigma[0,0]))
    loglike.append(l)

    # Update
    new_state = x_ + Kt @ epsilun
    xk[:, i] = new_state[:,0]
    P = (np.eye(2) - Kt @ H) @ P_  # updated estimate covariance

    x_s.append(x_)
    P_s.append(P_)
    x_updates.append(new_state)
    P_updates.append(P)

negloglike = np.sum(loglike)
x_s = np.hstack(x_s)
x_updates = np.hstack(x_updates)


plt.plot(x_s[1,:])
plt.plot(x_updates[1,:])
plt.plot(T_in, color="darkred")



step = 1
x = x_s[1,step]
v = P_s[step][1,1]

x = x_updates[1,step]
v = P_updates[step][1,1]

# Given from your Kalman filter:
mu = x        # mean
sigma2 = v    # variance
sigma = np.sqrt(sigma2)

# Create x-axis values around the mean (±3σ covers 99.7%)
xs = np.linspace(mu - 3*sigma, mu + 3*sigma, 400)

# Gaussian pdf
pdf = (1 / np.sqrt(2*np.pi*sigma2)) * np.exp(-(xs - mu)**2 / (2*sigma2))

# Plot
plt.plot(xs, pdf)
plt.title("Gaussian Distribution from Kalman Filter Mean and Variance")
plt.xlabel("State value")
plt.ylabel("Probability density")
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt

# example time horizon
T = 10


step = [1, 20]
preds_mean = x_s[1,step[0]:step[1]]
preds_var = np.array(P_s[step[0]:step[1]]).flatten()[3::4]

update_mean = x_updates[1,step[0]:step[1]]
update_var = np.array(P_updates[step[0]:step[1]]).flatten()[3::4]

# x-axis grid for PDFs
x = np.linspace(preds_mean - 3 * preds_var**0.5, preds_mean + 3 * preds_var**0.5, 500)

plt.figure(figsize=(10, 8))
scale = 1

for k in range(T):
    # prediction pdf
    pdf_pred = (1 / np.sqrt(2*np.pi*preds_var[k])) * np.exp(-(x - preds_mean[k])**2 / (2 * preds_var[k]))
    # update pdf
    pdf_upd  = (1 / np.sqrt(2*np.pi*update_var[k]))  * np.exp(-(x - update_mean[k])**2  / (2 * update_var[k]))

    # vertical offset
    y_offset = k * scale

    plt.plot(x, pdf_pred + y_offset, label="prediction" if k==0 else "", alpha=0.8)
    # plt.plot(x, pdf_upd  + y_offset, label="update" if k==0 else "", alpha=0.8)

plt.xlabel("State value")
plt.ylabel("Time step")
plt.title("Timeline of 1D Gaussian Distributions")
plt.legend()
plt.grid(True)
plt.show()










