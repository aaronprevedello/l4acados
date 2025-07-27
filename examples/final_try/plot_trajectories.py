import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from utils import visualize_inverted_pendulum

# === PARSING ARGOMENTI DA TERMINALE ===
parser = argparse.ArgumentParser(description="Plot dati simulazione con riferimento opzionale.")
parser.add_argument("--ref", type=str, default=None, help="Percorso al file .npz contenente i dati di riferimento")
args = parser.parse_args()

# === CONFIG ===
folder = "grafici"  # cartella dove hai salvato i .npz

# === CARICA TUTTI I FILE .npz ORDINATI ===
files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
if not files:
    raise FileNotFoundError(f"Nessun file .npz trovato nella cartella '{folder}'")

X = []
U = []
T = []
p_pred = []
theta_pred = []
v_pred = []
omega_pred = []

for f in files:
    data = np.load(os.path.join(folder, f))
    X.append(data["x"])
    U.append(data["u"])
    T.append(data["i"])
    p_pred.append(data["p_predicted"])    
    theta_pred.append(data["theta_predicted"])
    v_pred.append(data["v_predicted"])
    omega_pred.append(data["omega_predicted"])
    #print("X shape is ", data["x"].shape)

# X[0] = np.array([0.0, np.pi, 0.0, 0.0, 0.0])
#print("First item in X is ", X[0])
X = np.array(X)  # shape: (N_iter, dim_x)
U = np.array(U)  # shape: (N_iter, dim_u)
T = np.array(T)  # shape: (N_iter,)
ts_real = np.mean(np.diff(T))
p_pred = np.array(p_pred)
theta_pred = np.array(theta_pred)
v_pred = np.array(v_pred)
omega_pred = np.array(omega_pred)

print(f"Caricati {len(files)} step.")
print(f"Dimensione stato: {X.shape[1]}, Dimensione ingresso: {U.shape[1]}")

# === LOAD DEL REFERENCE ===
X_ref = None
T_ref = None
if args.ref is not None:
    if not os.path.exists(args.ref):
        raise FileNotFoundError(f"File di riferimento '{args.ref}' non trovato.")
    ref_data = np.load(args.ref)
    cart_ref = ref_data["cart_ref"]
    theta_ref = ref_data["theta_ref"]
    print(f"Caricato file di riferimento '{args.ref}' ")


# === PLOT STATI ===
# plt.figure(figsize=(10, 6))
# for i in range(4):
#     plt.plot(T, X[:, i], label=f"x[{i}]")
# plt.xlabel("Tempo")
# plt.ylabel("Stato")
# plt.title("Evoluzione degli stati")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# === PLOT INGRESSI ===
# plt.figure(figsize=(8, 4))
# for i in range(U.shape[1]):
#     plt.plot(T, U[:, i], label=f"u[{i}]")
# plt.xlabel("Tempo")
# plt.ylabel("Ingresso")
# plt.title("Ingressi nel tempo")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

n_horiz_steps = p_pred.shape[1]
t_horiz_mpc = np.linspace(0, (n_horiz_steps - 1) * 0.025, n_horiz_steps)

# === PLOT STATI E PASSATI===
plt.figure()
plt.subplot(2, 2, 1)
for i in [0]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
for i in range(0, len(T), 10):
    plt.plot(i*0.025 + t_horiz_mpc, p_pred[i, :], 'r--', alpha = 0.5, label='Pred cart pose' if i==0 else '')
if args.ref is not None:
    plt.plot(T, cart_ref[:len(T)], label= "cart reference")
plt.xlabel("Tempo [s]")
plt.ylabel("Posizione [m]")
plt.title("Cart Position")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2, 2, 2)
for i in [1]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
for i in range(0, len(T), 10):
    plt.plot(i*0.025 + t_horiz_mpc, theta_pred[i, :], 'g--', alpha = 0.5, label='Pred pendulum pose' if i==0 else '')
if args.ref is not None:
    plt.plot(T, theta_ref[:len(T)], label = "Pendulum references")
plt.xlabel("Tempo [s]")
plt.ylabel("Posizione [rad]")
plt.title("Angolo Pendolo")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2, 2, 3)
for i in [2]:
    plt.plot(T, X[:, i], label=f"v_cart")
for i in [3]:
    plt.plot(T, X[:, i], label=f"w_pend")
plt.xlabel("Tempo [s]")
plt.ylabel("Velocità [m/s] [rad/s]")
plt.title("Velocità cart")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2, 2, 4)
#for i in [4]:
#    plt.plot(T, X[:, i], label=f"u_act")
plt.plot(T, U, label = "u_des")
if X.shape[1] >=5:
    plt.plot(T, X[:, 4], label='u_act')
if X.shape[1] >=6:
    plt.plot(T, X[:, 5], label='u_past')
plt.xlabel("Tempo [s]")
plt.ylabel("Forza [N]")
plt.title("Ingressi")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === MOSTRA TUTTI I GRAFICI ===
plt.show()

if args.ref is not None:
    visualize_inverted_pendulum(X, U, T, ts_real, REF = cart_ref)
else:
    visualize_inverted_pendulum(X, U, T, ts_real)