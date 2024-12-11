
# Gradient information file description

In the .txt obtained after training, each two lines are the gradient vectors of the same parameter at the same training step (one line is discrete diffusion and the other line is continuous diffusion), in the format of pytorch tensor. After *grads_angle_op* is processed, the *result_storage_path/reconstructed_grad* folder contains multiple epochs. Each epoch folder contains tensor vectors of multiple parameters at several training steps, still in groups of two lines. The final result is that each parameter in the *result_storage_path/angle_param* file under each epoch contains an angle value in degrees.
