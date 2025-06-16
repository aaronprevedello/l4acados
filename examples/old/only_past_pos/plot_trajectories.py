import numpy as np
import os
import matplotlib.pyplot as plt

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

X = np.array(X)  # shape: (N_iter, dim_x)
U = np.array(U)  # shape: (N_iter, dim_u)
T = np.array(T)  # shape: (N_iter,)

print(f"Caricati {len(files)} step.")
print(f"Dimensione stato: {X.shape[1]}, Dimensione ingresso: {U.shape[1]}")

# === PLOT STATI ===
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(T, X[:, i], label=f"x[{i}]")
plt.xlabel("Tempo")
plt.ylabel("Stato")
plt.title("Evoluzione degli stati")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === PLOT INGRESSI ===
plt.figure(figsize=(8, 4))
for i in range(U.shape[1]):
    plt.plot(T, U[:, i], label=f"u[{i}]")
plt.xlabel("Tempo")
plt.ylabel("Ingresso")
plt.title("Ingressi nel tempo")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === PLOT STATI E PASSATI===
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
for i in [0, 4, 6]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
plt.xlabel("Tempo [s]")
plt.ylabel("Posizione [m]")
plt.title("Cart Position")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2, 1, 2)
for i in [1, 5, 7]:
    plt.plot(T, X[:, i], label=f"x[{i}]")
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

# === MOSTRA TUTTI I GRAFICI ===
plt.show()
