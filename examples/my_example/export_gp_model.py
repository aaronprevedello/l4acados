import numpy as np
import torch
import os

def export_gpytorch_model_to_c(model, output_dir, model_name="gp_cartpole"):
    """
    Exports a BatchIndependentMultitaskGPModel trained in GPyTorch to C code (ready for acados).

    Args:
        model: Trained GPyTorch model.
        output_dir: Folder to save .c and .h files.
        model_name: Base name for files and function prefixes.
    """

    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Extract parameters
    train_x = model.train_inputs[0].detach().cpu().numpy()   # (N, D)
    train_y = model.train_targets.detach().cpu().numpy()     # (N, P) if multitask
    N, D = train_x.shape
    P = train_y.shape[1]  # number of outputs

    covar_module = model.covar_module
    base_kernel = covar_module.base_kernel
    outputscale = covar_module.outputscale.detach().cpu().numpy()  # (P,)
    lengthscales = base_kernel.lengthscale.detach().cpu().numpy()  # (P, 1, D)
    noise = model.likelihood.noise.detach().cpu().numpy()          # (P,)

    # Precompute (K + noise I)^-1 y for each output
    alpha = []
    train_x_tensor = torch.tensor(train_x)
    noise_val = noise[0]  # shared noise for all outputs
    for p in range(P):
        ls = base_kernel.lengthscale[p,0,:]
        oscale = covar_module.outputscale[p]
    
        diff = train_x_tensor.unsqueeze(1) - train_x_tensor.unsqueeze(0)  # (N, N, D)
        dist_sq = (diff / ls).pow(2).sum(-1)  # (N, N)
        K = oscale * torch.exp(-0.5 * dist_sq)
    
        K += torch.eye(N) * noise_val  # use shared noise
    
        y = torch.tensor(train_y[:,p])
        alpha_p = torch.linalg.solve(K, y)
        alpha.append(alpha_p.detach().cpu().numpy())

    
    alpha = np.array(alpha)  # (P, N)

    # --- Write C files ---

    # Header
    header = f"""#ifndef {model_name.upper()}_H_
#define {model_name.upper()}_H_

#ifdef __cplusplus
extern "C" {{
#endif

#define INPUT_DIM {D}
#define OUTPUT_DIM {P}
#define NUM_TRAINING_POINTS {N}

int {model_name}_dynamics(const double** in, double** out, int* iw, double* w, void *mem);

#ifdef __cplusplus
}}
#endif

#endif // {model_name.upper()}_H_
"""
    with open(os.path.join(output_dir, f"{model_name}.h"), "w") as f:
        f.write(header)

    # Source
    source = f"""#include "{model_name}.h"
#include <math.h>

// Training inputs
const double train_x[NUM_TRAINING_POINTS][INPUT_DIM] = {{
"""
    for x in train_x:
        source += "    {" + ", ".join([f"{xi:.16e}" for xi in x]) + "},\n"
    source += "};\n\n"

    # Alpha
    source += "// Alpha coefficients (precomputed)\n"
    source += "const double alpha[OUTPUT_DIM][NUM_TRAINING_POINTS] = {\n"
    for p in range(P):
        source += "    {" + ", ".join([f"{ai:.16e}" for ai in alpha[p]]) + "},\n"
    source += "};\n\n"

    # Lengthscales and outputscales
    source += "// Lengthscales\n"
    source += "const double lengthscale[OUTPUT_DIM][INPUT_DIM] = {\n"
    for p in range(P):
        source += "    {" + ", ".join([f"{l:.16e}" for l in lengthscales[p,0,:]]) + "},\n"
    source += "};\n\n"

    source += "// Outputscales\n"
    source += "const double outputscale[OUTPUT_DIM] = {"
    source += ", ".join([f"{o:.16e}" for o in outputscale])
    source += "};\n\n"

    # RBF kernel
    source += """
double rbf_kernel(int out_dim, const double* x1, const double* x2) {
    double sqdist = 0.0;
    for (int d = 0; d < INPUT_DIM; d++) {
        double diff = (x1[d] - x2[d]) / lengthscale[out_dim][d];
        sqdist += diff * diff;
    }
    return outputscale[out_dim] * exp(-0.5 * sqdist);
}

// GP prediction
void gp_predict(const double* xu, double* delta_x) {
    for (int out = 0; out < OUTPUT_DIM; out++) {
        double mean = 0.0;
        for (int i = 0; i < NUM_TRAINING_POINTS; i++) {
            double k = rbf_kernel(out, xu, train_x[i]);
            mean += k * alpha[out][i];
        }
        delta_x[out] = mean;
    }
}

// External function for acados
int {model_name}_dynamics(const double** in, double** out, int* iw, double* w, void *mem) {
    const double *xu = in[0];
    double *x_next = out[0];

    double delta_x[OUTPUT_DIM];

    // Predict GP correction
    gp_predict(xu, delta_x);

    // Baseline model (simple Euler forward model example)
    // NOTE: Replace this with your real base dynamics if needed
    double dt = 0.05;

    x_next[0] = xu[0] + dt * xu[1]; // x + dt * v
    x_next[1] = xu[1] + dt * (xu[4] - 0.1 * xu[1]); // v + dt * (u - damping)
    x_next[2] = xu[2] + dt * xu[3]; // theta + dt * theta_dot
    x_next[3] = xu[3] + dt * (9.81 * sin(xu[2]) + xu[4]); // theta_dot + dt * torque effect

    // Add GP correction
    for (int i = 0; i < OUTPUT_DIM; i++) {
        x_next[i] += delta_x[i];
    }

    return 0;
}
"""
    with open(os.path.join(output_dir, f"{model_name}.c"), "w") as f:
        f.write(source)

    print(f"Exported C files to {output_dir}")
    import time 
    time.sleep(5.5)

# Usage:
# from your script
# export_gpytorch_model_to_c(trained_model, "./c_generated_code", "cartpole_gp")
