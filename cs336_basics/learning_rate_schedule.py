import math

def get_lr_cosine_schedule(
        t: int,
        alpha_max: float,
        alpha_min: float,
        t_w: int,
        t_c: int
) -> float:
    if t < t_w:
        return t/t_w * alpha_max
    elif t <= t_c:
        return alpha_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min