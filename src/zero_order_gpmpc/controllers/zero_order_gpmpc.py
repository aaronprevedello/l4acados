import numpy as np
from .zoro_acados_utils import *
from .residual_learning_mpc import ResidualLearningMPC
from zero_order_gpmpc.models import ResidualModel


class ZeroOrderGPMPC(ResidualLearningMPC):
    def __init__(
        self,
        ocp,
        B=None,
        gp_model: ResidualModel = None,
        use_cython=True,
        path_json_ocp="zoro_ocp_solver_config.json",
        path_json_sim="zoro_sim_solver_config.json",
        build_c_code=True,
    ):
        # Set up all member variables but don't build the code yet. We still need to setup
        # the custom update.
        super().__init__(
            ocp,
            B=B,
            residual_model=gp_model,
            use_cython=use_cython,
            path_json_ocp=path_json_ocp,
            path_json_sim=path_json_sim,
            build_c_code=False,
        )

        self.setup_custom_update()

        self.build_c_code_done = False
        if build_c_code:
            self.build(
                use_cython=use_cython,
                build_c_code=build_c_code,
                path_json_ocp=path_json_ocp,
                path_json_sim=path_json_sim,
            )

    def solve(self):
        for i in range(self.ocp_opts["nlp_solver_max_iter"]):
            self.preparation()
            status_cupd = self.do_custom_update()
            status_feed = self.feedback()

            # ------------------- Check termination --------------------
            # check on residuals and terminate loop.
            residuals = self.ocp_solver.get_residuals()
            print("residuals after ", i, "SQP_RTI iterations:\n", residuals)

            if status_feed != 0:
                raise Exception(
                    "acados self.ocp_solver returned status {} in time step {}. Exiting.".format(
                        status_feed, i
                    )
                )

            if np.all(residuals < self.ocp_opts_tol_arr):
                break

    def setup_custom_update(self):
        template_c_file = "custom_update_function_zoro_template.in.c"
        template_h_file = "custom_update_function_zoro_template.in.h"
        custom_c_file = "custom_update_function.c"
        custom_h_file = "custom_update_function.h"

        # custom update: disturbance propagation
        self.ocp.solver_options.custom_update_filename = custom_c_file
        self.ocp.solver_options.custom_update_header_filename = custom_h_file

        self.ocp.solver_options.custom_templates = [
            (
                template_c_file,
                custom_c_file,
            ),
            (
                template_h_file,
                custom_h_file,
            ),
        ]

        self.ocp.solver_options.custom_update_copy = False
        """NOTE(@naefjo): As far as I understand you need to set this variable to True if you just
        want to copy an existing custom_update.c/h into the export directory and to False if you want
        to render the custom_udpate files from the template"""

        if self.ocp.zoro_description.input_P0_diag:
            self.zoro_input_P0 = np.diag(self.ocp.zoro_description.P0_mat)
        else:
            self.zoro_input_P0 = self.ocp.zoro_description.P0_mat

        self.zoro_input_W_diag = np.diag(self.ocp.zoro_description.W_mat)
        self.covariances_array = np.zeros(((self.N + 1) * self.nx**2,))

    def do_custom_update(self) -> None:
        """performs the acados custom update and propagates the covariances for the constraint tightening

        The array which is passed to the custom update function consists of an input array and
        an output array [cov_in, cov_out], where
        cov_in = [Sigma_x0, Sigma_w, [Sigma_GP_i forall i in (0, N-1)]] and
        cov_out = [-1*ones(3 * (N + 1)))] is a placeholder for the positional covariances used for
        visualization.

        Note that the function currently only supports setting the diagonal elements of the covariance matrices
        in the solver.
        """
        if not self.has_residual_model:
            return

        covariances_in = np.concatenate(
            (
                self.zoro_input_P0,
                self.zoro_input_W_diag,
                # self.residual_model.current_variance is updated with value_and_jacobian() call in preparation phase
                self.residual_model.current_variance.flatten(),
            )
        )
        covariances_in_len = covariances_in.size
        out_arr = np.ascontiguousarray(
            np.concatenate((covariances_in, -1.0 * np.ones(self.nx**2 * (self.N + 1))))
        )
        self.ocp_solver.custom_update(out_arr)
        self.covariances_array[:] = out_arr[covariances_in_len:]

        return 0
