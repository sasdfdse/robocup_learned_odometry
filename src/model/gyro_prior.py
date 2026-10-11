DTHETA_PRIOR = "gyro_trapezoid"


def gyro_dtheta(window, ang_vel_scale, step_dt):
    """Heading change over the last control step from the gyro.

    Trapezoid of the last two yaw-rate samples. It is ~2.6x more accurate than the
    plain OdomModel dtheta, because the history encoder blurs the latest frames, so
    the model only learns the residual on top of it.

    window: (B, T, D) scaled odometry inputs with ang_vel at [3:6].
    """
    return window[:, -2:, 5].mean(dim=1) / ang_vel_scale * step_dt
