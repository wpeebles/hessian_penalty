"""
Functions to smooth-out interpolations performed in Z-space. These are based on
Grant Sanderson's corresponding functions in manim. These are only used for
the purpose of visualization; they do not affect training.
"""

import torch


def smooth(t):
    error = torch.sigmoid(torch.tensor(-5.))
    return torch.clamp(torch.sigmoid(10 * (t - 0.5) - error) / (1 - 2 * error), 0, 1)


def there_and_back(t):
    new_t = torch.where(t < 0.5, 2 * t, 2 * (1 - t))
    return smooth(new_t)


def mid_right_mid_left_mid(steps, round=False):
    t = torch.linspace(0.0, 1.0, steps)
    ltr = there_and_back(t)
    left_to_mid_to_left = ltr / 2
    mid_to_right_to_mid = left_to_mid_to_left + 0.5
    mid_to_left = torch.flip(left_to_mid_to_left[:steps//2], (0,))
    mid_to_left_to_mid = torch.cat([mid_to_left, torch.flip(mid_to_left, (0,))])
    out = torch.flip(torch.cat([mid_to_right_to_mid, mid_to_left_to_mid]), (0,))
    if round:  # [0, steps-1]
        out = out.mul(steps - 1).round().long()
    else:  # [-1, 1]
        out = out.add(-0.5).mul(2)
    return out


def left_to_right(steps, round=False):
    t = torch.linspace(0.0, 1.0, steps)
    out = there_and_back(t)
    if round:
        out.mul_(steps - 1).round().long()
    else:
        out.add_(-0.5).mul_(2)
    return out
