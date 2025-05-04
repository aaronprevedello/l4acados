#ifndef GP_DYNAMICS_H_
#define GP_DYNAMICS_H_

#ifdef __cplusplus
extern "C" {
#endif

int gp_dynamics(const real_t** in, real_t** out, int* iw, real_t* w, void *mem);

#ifdef __cplusplus
}
#endif

#endif  // GP_DYNAMICS_H_
