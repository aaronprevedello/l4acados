#include "acados/utils/types.h"

int gp_dynamics(const real_t** in, real_t** out, int* iw, real_t* w, void *mem) {
    const real_t* xu = in[0];  // x = xu[0:4], u = xu[4]
    real_t* x_next = out[0];

    double dt = 0.05;
    // Simple dummy model; replace with your GP export
    x_next[0] = xu[0] + dt * xu[1];        // cart position
    x_next[1] = xu[1] + dt * xu[4];        // cart velocity
    x_next[2] = xu[2] + dt * xu[3];        // pole angle
    x_next[3] = xu[3] + dt * (-9.81 * xu[2]); // angular velocity

    return 0;
}
