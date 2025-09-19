import torch
import torch.nn.functional as F

from compressarc.cellgru.cellgru import CellGRU


torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')


class ARCCompressor:
    """
    CellGRU backbone shim that mimics the old ARCCompressor API so that
    train.py / solve_task.py / solution_selection.py keep working.

    It:
      - builds per-example input tokens with pad=0 (OOB), colors shifted +1
      - runs CellGRU for K recurrent steps
      - returns logits shaped like the original model:
        [example, color_no_black, x, y, in_out], where in_out=0(input),1(output)
      - returns fixed masks and dummy KL arrays for logger compatibility
    """

    def __init__(
        self,
        task,
        steps: int = 16,
        units: int = 128,
        kernel: int = 3,
        neighborhood: str = "moore",
        cell_type: str = "dcgru",
        dropout_keep_prob: float = 1.0,
        attn_heads: int = 8,
        num_latents: int = 64,
        mixer_type: str = "full",
        mixer_every: int = 4,
        mixer_heads: int = 8,
        mixer_depth: int = 4,
        mixer_pool_stride: int = 2,
        toroidal: bool = False,
        conditioning: str = "none",
        force_input_plane: bool = False,
        input_plane_index: int = 0,
    ):
        self.task = task
        self.steps = int(steps)

        # Task-local vocabulary:
        #   pad=0     (OOB canvas)
        #   colors=[0..n_colors] shifted by +1 -> tokens 1..(n_colors+1)
        # NOTE: task.n_colors excludes black; len(task.colors) == n_colors+1 (includes black=0)
        self.n_vocab = len(task.colors) + 1  # + pad

        # A plain DCGRU2D model with no controller/prompt bells & whistles
        self.net = CellGRU(
            num_units=units,
            n_input=self.n_vocab,
            n_classes=self.n_vocab,
            dropout_keep_prob=dropout_keep_prob,
            kernel_size=kernel,
            neighborhood=neighborhood,
            cell_type=cell_type,
            attn_heads=attn_heads,
            num_latents=num_latents,
            toroidal=toroidal,
            conditioning=conditioning,
            mixer_type=mixer_type,
            mixer_every=mixer_every,
            mixer_heads=mixer_heads,
            mixer_depth=mixer_depth,
            mixer_pool_stride=mixer_pool_stride,
            force_input_plane=force_input_plane,
            input_plane_index=input_plane_index,
        )

        # Make optimizer creation code in train/solve_task happy
        self.weights_list = list(self.net.parameters())

        # Dummy attributes so analyze_example.py won't explode if imported
        self.multiposteriors = {}
        self.decode_weights = {}
        self.target_capacities = {}

    # --- helpers -------------------------------------------------------------

    def _build_input_tokens(self):
        """
        Returns x_in: [B,H,W] with pad=0 outside example bounds,
        and tokens 1.. for in-bounds colors (index within task.colors, not raw ARC color id).
        """
        prob = self.task.problem           # [B,H,W,2] ints in 0..n_colors (index within task.colors)
        masks = self.task.masks            # [B,H,W,2] float {0,1}
        B, H, W = prob.shape[0], self.task.n_x, self.task.n_y

        inp_idx = prob[:, :, :, 0].long()              # [B,H,W]  (input grids)
        inb      = (masks[:, :, :, 0] > 0.5)           # [B,H,W]  in-bounds mask

        tokens = torch.zeros((B, H, W), dtype=torch.long, device=prob.device)
        tokens = torch.where(inb, inp_idx + 1, tokens) # pad=0, black -> token=1, etc.
        return tokens

    def _fixed_masks(self, device, dtype):
        """
        Builds x_mask, y_mask shaped [B, H, 2], [B, W, 2] with 1.0 inside known sizes and 0 outside.
        This plays nicely with CompressARC's crop scorer.
        """
        B, H, W = self.task.n_examples, self.task.n_x, self.task.n_y
        x_mask = torch.zeros((B, H, 2), device=device, dtype=dtype)
        y_mask = torch.zeros((B, W, 2), device=device, dtype=dtype)
        for e in range(B):
            H_in, W_in = self.task.shapes[e][0]
            H_out, W_out = self.task.shapes[e][1]
            x_mask[e, :H_in, 0] = 1.0; y_mask[e, :W_in, 0] = 1.0   # input side
            x_mask[e, :H_out, 1] = 1.0; y_mask[e, :W_out, 1] = 1.0 # output side
        return x_mask, y_mask

    # --- the API expected by train.py ---------------------------------------

    def forward(self):
        """
        Returns:
          logits   : [B, color_no_black, H, W, 2]
          x_mask   : [B, H, 2]
          y_mask   : [B, W, 2]
          KL_amounts: list[Tensor] (dummy/regularizer)
          KL_names  : list[str]
        """
        B, H, W = self.task.n_examples, self.task.n_x, self.task.n_y

        # Build tokens and run recurrent dynamics
        x_in = self._build_input_tokens()                 # [B,H,W]
        logits_bhwc, sat_loss = self.net(x_in, steps=self.steps)  # [B,H,W,n_vocab]

        # Map to CompressARC shapes.
        # Drop pad=0 and also drop black class (token=1). We keep COLOR_NO_BLACK channels
        # because train.take_step() prepends the black channel itself.
        logits_no_black = logits_bhwc[..., 2:]            # [B,H,W,C_nb], C_nb = n_colors (=len(colors)-1)
        out_logits = logits_no_black.permute(0, 3, 1, 2)  # [B,C_nb,H,W]

        # Channel 0 (input) needs something; give a confident one-hot of the input colors.
        # Targets are indices in 0..n_colors; shift so 1.. maps to 0..C_nb-1; black (0) -> no channel.
        C_nb = out_logits.shape[1]
        inp_idx = self.task.problem[:, :, :, 0].long()       # [B,H,W] in 0..n_colors
        adj = torch.clamp(inp_idx - 1, min=0)                # 0->0 (we'll zero it below), others -> c-1
        in_onehot = F.one_hot(adj, num_classes=C_nb).permute(0, 3, 1, 2).to(out_logits.dtype)
        in_onehot = in_onehot * (inp_idx > 0).unsqueeze(1).to(out_logits.dtype)
        inp_logits = in_onehot * 10.0                        # sharp but stable

        # Stack in_out axis as last dim
        logits = torch.stack([inp_logits, out_logits], dim=-1)  # [B,C_nb,H,W,2]

        # Simple deterministic masks
        x_mask, y_mask = self._fixed_masks(device=logits.device, dtype=logits.dtype)

        # Dummy KL for the logger (we can reuse saturation as a tiny reg term)
        KL_amounts = [sat_loss.reshape(1)]
        KL_names   = ["(0,0,0,0,0)"]

        return logits, x_mask, y_mask, KL_amounts, KL_names
