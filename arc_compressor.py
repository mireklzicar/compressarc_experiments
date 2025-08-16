import numpy as np
import torch

import initializers
import layers


np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')


def sample_op_st(pi_logits, tau=1.0):
    g = -torch.log(-torch.log(torch.rand_like(pi_logits).clamp_min(1e-9)) + 1e-9)
    y_soft = torch.softmax((pi_logits + g) / tau, dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(-1, keepdim=True), 1.0)
    return (y_hard - y_soft).detach() + y_soft


class ARCCompressor:
    """
    The main model class for the VAE Decoder in our solution to ARC.
    """

    # Define the channel dimensions that all the layers use
    n_layers = 4
    share_up_dim = 16
    share_down_dim = 8
    decoding_dim = 4
    softmax_dim = 2
    cummax_dim = 4
    shift_dim = 4
    nonlinear_dim = 16

    # This function gives the channel dimension of the residual stream depending on
    # which dimensions are present, for every tensor in the multitensor.
    def channel_dim_fn(self, dims):
        return 16 if dims[2] == 0 else 8

    def __init__(self, task, use_dsl=True, use_gumbel_softmax=False, temperature=1.0):
        """
        Create a model that is tailored to the given task, and initialize all the weights.
        The weights are symmetrized such that swapping the x and y dimension ordering should
        make the output's dimension ordering also swapped, for the same weights. This may not
        be exactly correct since symmetrizing all operations is difficult.
        Args:
            task (preprocessing.Task): The task which the model is to be made for solving.
            use_dsl (bool): Whether to use the differentiable DSL.
        """
        self.multitensor_system = task.multitensor_system
        self.use_gumbel_softmax = use_gumbel_softmax
        self.temperature = temperature

        # Initialize weights
        initializer = initializers.Initializer(self.multitensor_system, self.channel_dim_fn)

        self.multiposteriors = initializer.initialize_multiposterior(self.decoding_dim)
        self.decode_weights = initializer.initialize_multilinear([self.decoding_dim, self.channel_dim_fn])
        initializer.symmetrize_xy(self.decode_weights)
        self.target_capacities = initializer.initialize_multizeros([self.decoding_dim])

        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights = []
        self.shift_weights = []
        self.direction_share_weights = []
        self.nonlinear_weights = []

        for layer_num in range(self.n_layers):
            self.share_up_weights.append(initializer.initialize_multiresidual(self.share_up_dim, self.share_up_dim))
            self.share_down_weights.append(initializer.initialize_multiresidual(self.share_down_dim, self.share_down_dim))
            output_scaling_fn = lambda dims: self.softmax_dim * (2 ** (dims[1] + dims[2] + dims[3] + dims[4]) - 1)
            self.softmax_weights.append(initializer.initialize_multiresidual(self.softmax_dim, output_scaling_fn))
            self.cummax_weights.append(initializer.initialize_multiresidual(self.cummax_dim, self.cummax_dim))
            self.shift_weights.append(initializer.initialize_multiresidual(self.shift_dim, self.shift_dim))
            self.direction_share_weights.append(initializer.initialize_multidirection_share())
            self.nonlinear_weights.append(initializer.initialize_multiresidual(self.nonlinear_dim, self.nonlinear_dim))

        self.head_weights = initializer.initialize_head()
        self.mask_weights = initializer.initialize_linear(
            [1, 0, 0, 1, 0], [self.channel_dim_fn([1, 0, 0, 1, 0]), 2]
        )

        # Symmetrize weights so that their behavior is equivariant to swapping x and y dimension ordering
        for weight_list in [
            self.share_up_weights,
            self.share_down_weights,
            self.softmax_weights,
            self.cummax_weights,
            self.shift_weights,
            self.nonlinear_weights,
        ]:
            for layer_num in range(self.n_layers):
                initializer.symmetrize_xy(weight_list[layer_num])

        for layer_num in range(self.n_layers):
            initializer.symmetrize_direction_sharing(self.direction_share_weights[layer_num])

        # --- DSL config ---
        self.dsl_K = 6            # number of ops in dispatcher
        self.dsl_steps = 6        # max steps to run
        self.dsl_d_model = 32     # pointer key/query dim

        # Global per-example heads (dims [1,0,0,0,0]):
        self.dsl_op_head = initializer.initialize_linear([1,0,0,0,0], [self.channel_dim_fn, self.dsl_K])
        self.dsl_shift_head = initializer.initialize_linear([1,0,0,0,0], [self.channel_dim_fn, 2])
        self.dsl_axis_head  = initializer.initialize_linear([1,0,0,0,0], [self.channel_dim_fn, 2])
        self.dsl_k_head     = initializer.initialize_linear([1,0,0,0,0], [self.channel_dim_fn, 4])

        # Colors (output size = n_colors):
        self.dsl_paint_color_head = initializer.initialize_linear(
            [1,0,0,0,0], [self.channel_dim_fn, lambda d: self.multitensor_system.n_colors]
        )
        self.dsl_src_color_head = initializer.initialize_linear(
            [1,0,0,0,0], [self.channel_dim_fn, lambda d: self.multitensor_system.n_colors]
        )
        self.dsl_dst_color_head = initializer.initialize_linear(
            [1,0,0,0,0], [self.channel_dim_fn, lambda d: self.multitensor_system.n_colors]
        )

        # Spatial heads (dims [1,0,0,1,1]) -> per-pixel mask and pointer Q/K
        self.dsl_mask_head = initializer.initialize_linear([1,0,0,1,1], [self.channel_dim_fn, 1])
        self.dsl_Q_head    = initializer.initialize_linear([1,0,0,1,1], [self.channel_dim_fn, self.dsl_d_model])
        self.dsl_K_head    = initializer.initialize_linear([1,0,0,1,1], [self.channel_dim_fn, self.dsl_d_model])

        # Optional halting head (per-step): scalar per example
        self.use_dsl = use_dsl
        if self.use_dsl:
            self.dsl_stop_head = initializer.initialize_linear([1,0,0,0,0], [self.channel_dim_fn, 1])
        self.weights_list = initializer.weights_list


    def _dsl_params_from_features(self, x, temperature=1.0):
        """
        Produce all op parameters from your multitensor features x.
        Returns dict with tensors ready for dsl.apply_mixture.
        """
        # Global (per-example) features
        if x[[1,0,0,0,0]] is None:
            # get the device and dtype from another tensor in x
            sample_tensor = next(t for t in x.data_values() if t is not None)
            shape = self.multitensor_system.shape([1,0,0,0,0], self.channel_dim_fn([1,0,0,0,0]))
            x[[1,0,0,0,0]] = torch.zeros(shape, device=sample_tensor.device, dtype=sample_tensor.dtype)

        g = x[[1,0,0,0,0]]  # [B, chan]

        # Spatial (per-pixel) features
        s = x[[1,0,0,1,1]]  # [B, X, Y, chan]

        # Heads
        pi_logits = layers.affine(g, self.dsl_op_head, use_bias=True)                     # [B,K]
        shift     = layers.affine(g, self.dsl_shift_head, use_bias=True)                  # [B,2]
        axis_lg   = layers.affine(g, self.dsl_axis_head,  use_bias=True)                  # [B,2]
        k_lg      = layers.affine(g, self.dsl_k_head,     use_bias=True)                  # [B,4]
        paint_col = layers.affine(g, self.dsl_paint_color_head, use_bias=True)            # [B,C]
        src_col   = layers.affine(g, self.dsl_src_color_head,   use_bias=True)            # [B,C]
        dst_col   = layers.affine(g, self.dsl_dst_color_head,   use_bias=True)            # [B,C]

        mask_logits = layers.affine(s, self.dsl_mask_head, use_bias=True)                 # [B,X,Y,1]
        Q = layers.affine(s, self.dsl_Q_head, use_bias=True)                              # [B,X,Y,D]
        K = layers.affine(s, self.dsl_K_head, use_bias=True)                              # [B,X,Y,D]

        # Reorder mask to [B,1,X,Y]
        mask_logits = mask_logits.permute(0,3,1,2).contiguous()

        if self.use_gumbel_softmax:
            pi = sample_op_st(pi_logits, tau=temperature)
        else:
            pi = torch.softmax(pi_logits, dim=-1)

        return {
            "pi": pi,
            "mask": mask_logits,
            "shift": shift,
            "axis_logits": axis_lg,
            "k_logits": k_lg,
            "paint_color": paint_col,
            "src_color": src_col,
            "dst_color": dst_col,
            "Q": Q,
            "K": K,
        }

    def _run_dsl(self, P0, x, P_in=None):
        """
        Iteratively apply K-op mixture steps to the canvas.
        P0: [B,C,X,Y] probs
        """
        import dsl  # local module

        P = P0
        halts = []
        for t in range(self.dsl_steps):
            params = self._dsl_params_from_features(x)
            if P_in is not None:
                params["P_in"] = P_in
            P = dsl.apply_mixture(P, params["pi"], params)
            # optional halting
            if hasattr(self, "dsl_stop_head"):
                g = x[[1,0,0,0,0]]
                stop_logit = layers.affine(g, self.dsl_stop_head, use_bias=True).squeeze(-1)  # [B]
                p_stop = torch.sigmoid(stop_logit)
                halts.append(p_stop)
                # you can early-break at inference if (p_stop>0.5).all()
        return P, halts

    def forward(self):
        """
        Compute the forward pass of the VAE decoder. Start by using internally stored latents,
        and process from there. Output an [example, color, x, y, channel] tensor for the colors,
        and an [example, x, channel] and [example, y, channel] tensor for the masks.
        Returns:
            Tensor: An [example, color, x, y, channel] tensor, where for every example,
                    input/output (picked by channel dimension), and every pixel (picked
                    by x and y dimensions), we have a vector full of logits for that
                    pixel being each possible color.
            Tensor: An [example, x, channel] tensor, where for every example, input/output
                    (picked by channel dimension), and every x, we assign a score that
                    contributes to the likelihood that that index of the x dimension is not
                    masked out in the prediction.
            Tensor: An [example, y, channel] tensor, used in the same way as above.
            list[Tensor]: A list of tensors indicating the amount of KL contributed by each component
                    tensor in the layers.decode_latents() step.
            list[str]: A list of tensor names that correspond to each tensor in the aforementioned output.
        """
        # Decoding layer
        x, KL_amounts, KL_names = layers.decode_latents(
            self.target_capacities, self.decode_weights, self.multiposteriors
        )

        for layer_num in range(self.n_layers):
            # Multitensor communication layer
            x = layers.share_up(x, self.share_up_weights[layer_num])

            # Softmax layer
            x = layers.softmax(x, self.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Directional layers
            x = layers.cummax(
                x, self.cummax_weights[layer_num], self.multitensor_system.task.masks,
                pre_norm=False, post_norm=True, use_bias=False
            )
            x = layers.shift(
                x, self.shift_weights[layer_num], self.multitensor_system.task.masks,
                pre_norm=False, post_norm=True, use_bias=False
            )

            # Directional communication layer
            x = layers.direction_share(x, self.direction_share_weights[layer_num], pre_norm=True, use_bias=False)

            # Nonlinear layer
            x = layers.nonlinear(x, self.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)

            # Multitensor communication layer
            x = layers.share_down(x, self.share_down_weights[layer_num])

            # Normalization layer
            x = layers.normalize(x)

        # Linear Heads
        output = (
            layers.affine(x[[1, 1, 0, 1, 1]], self.head_weights, use_bias=False)
            + 100 * self.head_weights[1]
        )
        x_mask = layers.affine(x[[1, 0, 0, 1, 0]], self.mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1, 0, 0, 0, 1]], self.mask_weights, use_bias=True)

        # -------- DSL refinement (new) --------
        if self.use_dsl:
            # Treat output[...,1] as the target canvas logits (channel=‘output’)
            # Convert to probs, run the program, convert back to logits.
            import dsl
            B, C, X, Y, two = output.shape
            assert two == 2, "expected last dim=2 (in/out)"

            logits0 = output[..., 1]                 # [B,C,X,Y]
            P0 = dsl.probs_from_logits(logits0)      # soft canvas

            # Optional: source canvas for pointer-copy (from the input channel 0)
            P_in = dsl.probs_from_logits(output[..., 0].detach())  # detach to keep it read-only

            P_refined, _halts = self._run_dsl(P0, x, P_in=P_in)
            logits_refined = dsl.logits_from_probs(P_refined)

            # write back
            output = torch.stack([output[...,0], logits_refined], dim=-1)

        # Postprocessing masks
        x_mask, y_mask = layers.postprocess_mask(self.multitensor_system.task, x_mask, y_mask)
        return output, x_mask, y_mask, KL_amounts, KL_names

