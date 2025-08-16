# dsl.py
import torch
import torch.nn.functional as F

def probs_from_logits(logits):
    # logits: [B,C,X,Y]
    return torch.softmax(logits, dim=1)

def logits_from_probs(P, eps=1e-8):
    return torch.log(P.clamp_min(eps))

# ---------- masks ----------
def predict_mask(mask_logits):  # [B,1,X,Y] logits -> [B,1,X,Y] in [0,1]
    return torch.sigmoid(mask_logits)

# ---------- core ops ----------
def op_paint(P, mask, color_logits):
    # P: [B,C,X,Y], mask: [B,1,X,Y], color_logits: [B,C]
    M = predict_mask(mask)
    c = torch.softmax(color_logits, dim=-1)[:, :, None, None]  # [B,C,1,1]
    return (1 - M)*P + M*c

def _affine_grid_from_shift(B, H, W, dx, dy, device):
    # dx,dy in pixels (float)
    tx = 2*dx/(W-1)
    ty = 2*dy/(H-1)
    theta = torch.zeros(B,2,3, device=device)
    theta[:,0,0] = 1; theta[:,1,1] = 1
    theta[:,0,2] = tx; theta[:,1,2] = ty
    return F.affine_grid(theta, size=(B,1,H,W), align_corners=True)

def op_translate(P, mask, shift):
    # shift: [B,2] -> (dx,dy)
    B,C,H,W = P.shape
    dx, dy = shift[:,0], shift[:,1]
    grid = _affine_grid_from_shift(B,H,W,dx,dy,P.device)
    moved = F.grid_sample(P, grid, mode='bilinear', padding_mode='zeros',
                          align_corners=True)
    M = predict_mask(mask)
    return (1 - M)*P + M*moved

def op_reflect(P, mask, axis_logits):
    # axis ∈ {vertical (flip W dim), horizontal (flip H dim)}
    gate = torch.softmax(axis_logits, dim=-1)  # [B,2]
    Pv = torch.flip(P, dims=[-1])  # vertical
    Ph = torch.flip(P, dims=[-2])  # horizontal
    Pref = gate[:,0,None,None,None]*Pv + gate[:,1,None,None,None]*Ph
    M = predict_mask(mask)
    return (1 - M)*P + M*Pref

def op_rot90(P, mask, k_logits):
    # mix over k ∈ {0,1,2,3}
    w = torch.softmax(k_logits, dim=-1)  # [B,4]
    B, C, H, W = P.shape
    max_dim = max(H, W)

    # Pad P to be square
    ph = max_dim - H
    pw = max_dim - W
    padded_P = F.pad(P, (0, pw, 0, ph))

    rots = []
    for k in range(4):
        # Rotate the padded tensor
        rotated_padded = padded_P.rot90(k, (-2, -1))
        # Crop it back to the original size
        cropped_rotated = rotated_padded[..., :H, :W]
        rots.append(cropped_rotated)

    R = torch.stack(rots, dim=1)  # [B,4,C,H,W]
    Pref = (w[:, :, None, None, None] * R).sum(1)
    M = predict_mask(mask)
    return (1 - M) * P + M * Pref

def op_replace_color(P, mask, src_color_logits, dst_color_logits, sharp=10.0):
    M = predict_mask(mask)
    src = torch.softmax(src_color_logits, dim=-1)[:, :, None, None]  # [B,C,1,1]
    dst = torch.softmax(dst_color_logits, dim=-1)[:, :, None, None]
    match = (P * src).sum(1, keepdim=True)                # [B,1,X,Y]
    A = torch.sigmoid(sharp*match)                         # soft selector
    Pnew = (1 - A)*P + A*dst
    return (1 - M)*P + M*Pnew

def op_pointer_copy(P, mask, Q, K, P_src=None):
    # Q,K: [B,X,Y,D]; P_src: [B,C,X,Y] (defaults to P)
    if P_src is None: P_src = P
    B,C,X,Y = P.shape
    Qf = Q.view(B, X*Y, -1)            # [B, HW, D]
    Kf = K.view(B, X*Y, -1).transpose(1,2)  # [B, D, HW]
    A = torch.softmax(Qf @ Kf / (Qf.shape[-1]**0.5), dim=-1)  # [B, HW, HW]
    Vf = P_src.permute(0,2,3,1).reshape(B, X*Y, C)            # [B, HW, C]
    Yf = A @ Vf                                               # [B, HW, C]
    Y = Yf.view(B, X, Y, C).permute(0,3,1,2)                  # [B,C,X,Y]
    M = predict_mask(mask)
    return (1 - M)*P + M*Y

# ---------- dispatcher (mixture over ops) ----------
def apply_mixture(P, pi, params):
    """
    P: [B,C,X,Y]
    pi: [B,K] mixture over ops in order: [paint, translate, reflect, rot90, replace, pointer]
    params: dict of tensors required by ops (already batched)
    """
    outs = []
    outs.append(op_paint(P, params["mask"], params["paint_color"]))                           # 0
    outs.append(op_translate(P, params["mask"], params["shift"]))                             # 1
    outs.append(op_reflect(P, params["mask"], params["axis_logits"]))                         # 2
    outs.append(op_rot90(P, params["mask"], params["k_logits"]))                              # 3
    outs.append(op_replace_color(P, params["mask"], params["src_color"], params["dst_color"]))# 4
    outs.append(op_pointer_copy(P, params["mask"], params["Q"], params["K"], params.get("P_in")))
    O = torch.stack(outs, dim=1)                           # [B,K,C,X,Y]
    return (pi[:, :, None, None, None] * O).sum(1)         # [B,C,X,Y]