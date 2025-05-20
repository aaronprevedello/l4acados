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

for f in files:
    data = np.load(os.path.join(folder, f))
    X.append(data["x"])
    U.append(data["u"])
    T.append(data["i"])
    #print("X shape is ", data["x"].shape)

# X[0] = np.array([0.0, np.pi, 0.0, 0.0, 0.0])
#print("First item in X is ", X[0])
X = np.array(X)  # shape: (N_iter, dim_x)
U = np.array(U)  # shape: (N_iter, dim_u)
T = np.array(T)  # shape: (N_iter,)

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

# === PLOT STATI E PASSATI===
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
for i in [0]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
plt.plot(T, cart_ref[:len(T)], label= "cart reference")
plt.xlabel("Tempo [s]")
plt.ylabel("Posizione [m]")
plt.title("Cart Position")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2, 1, 2)
for i in [1]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
plt.plot(T, theta_ref[:len(T)], label = "Pendulum references")
plt.xlabel("Tempo [s]")
plt.ylabel("Posizione [rad]")
plt.title("Angolo Pendolo")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
for i in [2]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
plt.xlabel("Tempo [s]")
plt.ylabel("Velocità [m/s]")
plt.title("Velocità cart")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2, 1, 2)
for i in [3]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
plt.xlabel("Tempo [s]")
plt.ylabel("Velocità [rad/s]")
plt.title("Velocità Pendolo")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 6))
for i in [4]:
    plt.plot(T, X[:, i], label=f"u_act")
plt.plot(T, U, label = "u_des")
plt.xlabel("Tempo [s]")
plt.ylabel("Forza [N]")
plt.title("Ingressi")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === MOSTRA TUTTI I GRAFICI ===
plt.show()

visualize_inverted_pendulum(X, U, T, REF = cart_ref)
