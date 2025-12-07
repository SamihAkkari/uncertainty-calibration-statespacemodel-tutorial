
import pandas as pd
import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import expm, inv
from matplotlib import pyplot as plt


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
        n = Ac.shape[0]
        foo = expm(np.block([[-Ac, Qc], [np.zeros(np.shape(Ac)), Ac.T]]) * time_step)
        Q = np.dot(foo[n:2 * n, n:2 * n].T, foo[0:n, n:2 * n])
        # Measurement error covariance matrix (discrete)
        R = Rc / time_step
        return F, G, Q, R


df = pd.read_csv("../kalman-filter/data.csv", delimiter=";").set_index("Step")
df.drop(df.index[-1], axis=0, inplace=True)
data = df[df['Month'].isin([1, 2, 3])]  ## keep only summer months
sT = 3600 * 24
data.index = data.index*3600 / sT  # Change time scale to days

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

r = 1e-2              # measurement noise for internal temp

Ac, Bc, H, Qc, Rc = build_continuous_matrices(R1, R2, C1, C2, qi, qe, r)
F, G, Q, R = discretize(time_step, Ac, Bc, Qc, Rc)


## Initialisation
uk = np.vstack([T_ext, q])
xk = np.zeros((F.shape[0], uk.shape[1]))  # initial state
P = np.eye(2)  # initial state
xk[:, 0] = np.array([T_w_initial, T_in_initial])  # initial external temp

# Initialize as lists to preserve dimensions when appending
x_s = [np.array([xk[:, i-1]]).T]
P_s = [P]
x_updates = [np.array([xk[:, i-1]]).T]
P_updates = []
loglikelihoods = [1]
likelihoods = [1]
sigmas_list = [1]

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

    likelihood = (1 / np.sqrt(2*np.pi*Sigma[0,0])) * np.exp(-(epsilun[0,0])**2 / (2 * Sigma[0,0]))
    loglikelihood = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(Sigma[0,0]) - 0.5 * (epsilun[0,0] ** 2 / Sigma[0,0])

    # l = 0.5 * (epsilun[0,0] ** 2 / Sigma[0,0] + np.log(Sigma[0,0]))
    loglikelihoods.append(loglikelihood)
    likelihoods.append(likelihood)

    # Update
    new_state = x_ + Kt @ epsilun
    xk[:, i] = new_state[:,0]
    P = (np.eye(2) - Kt @ H) @ P_  # updated estimate covariance

    x_s.append(x_)
    P_s.append(P_)
    x_updates.append(new_state)
    P_updates.append(P)
    sigmas_list.append(Sigma[0,0])

loglike = np.sum(loglikelihoods)
x_s = np.hstack(x_s)
x_updates = np.hstack(x_updates)
sigmas_array = np.array(sigmas_list)


plt.plot(x_s[1,:])
plt.plot(x_updates[1,:])
plt.plot(T_in, color="darkred")










step = 5

measured_mean = T_in[step]
measured_var = r ** 2
xm = np.linspace(measured_mean - 3 * measured_var**0.5, measured_mean + 3 * measured_var**0.5, 500)
pdf_meas = (1 / np.sqrt(2*np.pi*measured_var)) * np.exp(-(xm - measured_mean)**2 / (2 * measured_var))

preds_mean = x_s[1,step]
preds_var = np.array(P_s[step]).flatten()[3::4]
xp = np.linspace(preds_mean - 3 * preds_var[0]**0.5, preds_mean + 3 * preds_var[0]**0.5, 500)
pdf_pred = (1 / np.sqrt(2*np.pi*preds_var[0])) * np.exp(-(xp - preds_mean)**2 / (2 * preds_var[0]))


update_mean = x_updates[1,step]
update_var = np.array(P_updates[step]).flatten()[3::4]
xu = np.linspace(update_mean - 3 * update_var[0]**0.5, update_mean + 3 * update_var[0]**0.5, 500)
pdf_upd  = (1 / np.sqrt(2*np.pi*update_var[0]))  * np.exp(-(xu - update_mean)**2  / (2 * update_var[0]))

plt.figure(figsize=(10, 8))
plt.plot(xm, pdf_meas, label="measurement", alpha=0.8)
plt.plot(xp, pdf_pred, label="prediction", alpha=0.8)
plt.plot(xu, pdf_upd, label="update", alpha=0.8)

plt.xlabel("State value")
plt.ylabel("Density")
plt.title("Timeline of 1D Gaussian Distributions")
plt.legend()
plt.grid(True)
plt.show()

xl = np.linspace(preds_mean - 3 * sigmas_array[step]**0.5, preds_mean + 3 * sigmas_array[step]**0.5, 500)
pdf_likelihood = (1 / np.sqrt(2*np.pi*sigmas_array[step])) * np.exp(-(xl - preds_mean)**2 / (2 * sigmas_array[step]))
plt.figure(figsize=(10, 8))
plt.plot(xl, pdf_likelihood, label="likelihood")
plt.axvline(x=measured_mean, color='red', label="measurement mean")
plt.xlabel("State value")
plt.ylabel("Density")
plt.title("Timeline of 1D Gaussian Distributions")
plt.legend()
plt.grid(True)
plt.show()


xl = np.linspace(preds_mean - 3 * sigmas_array[step]**0.5, preds_mean + 3 * sigmas_array[step]**0.5, 500)
pdf_loglikelihood = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigmas_array[step]) - 0.5 * ((xl - preds_mean) ** 2 / sigmas_array[step])
plt.figure(figsize=(10, 8))
plt.plot(xl, pdf_loglikelihood, label="loglikelihood")
plt.axvline(x=measured_mean, color='red', label="measurement mean")
plt.xlabel("State value")
plt.ylabel("Density")
plt.title("Timeline of 1D Gaussian Distributions")
plt.legend()
plt.grid(True)
plt.show()



steps_to_plot = [5, 6, 7, 8]
offset_scale = 30

# Fixed colors
color_meas = "red"
color_pred = "blue"
color_upd  = "green"

plt.figure(figsize=(14, 10))

legend_handles = []

for i, step in enumerate(steps_to_plot):
    offset = i * offset_scale

    # --- MEASUREMENT ---
    measured_mean = T_in[step]
    measured_var = r ** 2
    xm = np.linspace(measured_mean - 3*np.sqrt(measured_var),
                     measured_mean + 3*np.sqrt(measured_var), 500)
    pdf_meas = (1 / np.sqrt(2*np.pi*measured_var)) \
               * np.exp(-(xm - measured_mean)**2 / (2*measured_var))

    # --- PREDICTION ---
    preds_mean = x_s[1, step]
    preds_var = np.array(P_s[step]).flatten()[3::4][0]
    xp = np.linspace(preds_mean - 3*np.sqrt(preds_var),
                     preds_mean + 3*np.sqrt(preds_var), 500)
    pdf_pred = (1 / np.sqrt(2*np.pi*preds_var)) \
               * np.exp(-(xp - preds_mean)**2 / (2*preds_var))

    # --- UPDATE ---
    update_mean = x_updates[1, step]
    update_var = np.array(P_updates[step]).flatten()[3::4][0]
    xu = np.linspace(update_mean - 3*np.sqrt(update_var),
                     update_mean + 3*np.sqrt(update_var), 500)
    pdf_upd = (1 / np.sqrt(2*np.pi*update_var)) \
              * np.exp(-(xu - update_mean)**2 / (2*update_var))

    # --- PLOT ---
    l1, = plt.plot(xm, pdf_meas + offset, color=color_meas, alpha=0.8)
    l2, = plt.plot(xp, pdf_pred + offset, color=color_pred, alpha=0.8)
    l3, = plt.plot(xu, pdf_upd  + offset, color=color_upd,  alpha=0.8)

# Use each label only once
legend_handles = [l1, l2, l3]

plt.legend(legend_handles, ["measurement", "prior", "posterior"])

plt.xlabel("State value (Indoor temperature °C)")
plt.ylabel("Time step")
plt.title("Evolution of Gaussian PDFs Across Steps")

# ---- Y-axis step labels (THIS IS WHAT YOU WANTED) ----
yticks_positions = [i * offset_scale for i in range(len(steps_to_plot))]
plt.yticks(yticks_positions, steps_to_plot)

plt.grid(True)
plt.show()






steps_to_plot = [5, 6, 7, 8]
offset_scale = 5  # vertical spacing

color_ll = "purple"
color_meas = "red"

plt.figure(figsize=(12, 10))

legend_handles_ll = None
legend_handles_meas = None

for i, step in enumerate(steps_to_plot):
    offset = i * offset_scale

    # --- LOG-LIKELIHOOD ---
    preds_mean = x_s[1, step]
    sigma = sigmas_array[step]

    xl = np.linspace(preds_mean - 3*np.sqrt(sigma),
                     preds_mean + 3*np.sqrt(sigma), 500)

    pdf_likelihood = (1 / np.sqrt(2*np.pi*sigma)) * np.exp(-(xl - preds_mean)**2 / (2 * sigma))

    # --- MEASUREMENT MEAN ---
    measured_mean = T_in[step]

    # --- PLOT LL (shifted vertically) ---
    ll_handle, = plt.plot(xl, pdf_likelihood + offset,
                          color=color_ll, alpha=0.85)

    # --- RESTRICTED VERTICAL LINE FOR MEASUREMENT ---
    meas_handle, = plt.plot(
        [measured_mean, measured_mean],
        [offset, offset + offset_scale],
        color=color_meas,
        alpha=0.8
    )

    # Save handles once for legend
    if legend_handles_ll is None:
        legend_handles_ll = ll_handle
        legend_handles_meas = meas_handle

# Legend (only once)
plt.legend([legend_handles_ll, legend_handles_meas],
           ["likelihood", "measurement mean"])

plt.xlabel("State value")
plt.ylabel("Time step")
plt.title("Stacked Likelihood Evolution Across Time")

# Put time-step labels at the offsets
yticks_positions = [i * offset_scale for i in range(len(steps_to_plot))]
plt.yticks(yticks_positions, steps_to_plot)

plt.grid(True)
plt.show()
