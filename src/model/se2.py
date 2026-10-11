# Copyright 2024 Booster Robotics Technology Co., Ltd. ("Booster Robotics")
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
# Taken from humanoid-soccer runner/odom.py (Odom.se2_relative, Odom.se2_compose).
# Modified for robocup_learned_odometry: turned the methods into module-level functions.
import torch


def se2_relative(x1, y1, theta1, x2, y2, theta2, normalize_angle=True):
    dx_w = x2 - x1
    dy_w = y2 - y1

    c1 = torch.cos(theta1)
    s1 = torch.sin(theta1)
    dx = c1 * dx_w + s1 * dy_w
    dy = -s1 * dx_w + c1 * dy_w

    dtheta = theta2 - theta1
    if normalize_angle:
        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))

    return dx, dy, dtheta


def se2_compose(x, y, theta, dx, dy, dtheta, normalize_angle=True):

    c = torch.cos(theta)
    s = torch.sin(theta)

    new_x = x + c * dx - s * dy
    new_y = y + s * dx + c * dy
    new_theta = theta + dtheta

    if normalize_angle:
        new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))

    return new_x, new_y, new_theta
