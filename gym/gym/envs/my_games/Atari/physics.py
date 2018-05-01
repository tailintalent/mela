import numpy as np


def get_physics(physics_settings, dt):
    physics_list = []
    for physics_name, physics_params in physics_settings:
        if physics_name == "linear":
            physics_class = Linear
        elif physics_name == "gravity":
            physics_class = Uniform_Gravity
        elif physics_name == "Brownian_force":
            physics_class = Brownian_Force
        elif physics_name == "drag":
            physics_class = Drag
        elif physics_name == "central_force":
            physics_class = Central_Force
        elif physics_name == "point_force_paddle":
            physics_class = Point_Force_Paddle
        elif physics_name == "pairwise_force":
            physics_class = Pairwise_Force
        else:
            raise Exception("physics_name {0} not recognized!".format(physics_name))
        physics = physics_class(params = physics_params, dt = dt)
        physics_list.append(physics)
    return physics_list



class Physics(object):
    def __init__(self, dt, params = {}):
        self.dt = dt
        self.params = params

    def is_in_domain(self, ball, **kwargs):
        in_domain = True
        if "domain" in self.params:
            (x_min, x_max), (y_min, y_max) = self.params["domain"]
            if (x_min is not None and ball.x < x_min) or \
               (x_max is not None and ball.x > x_max) or \
               (y_min is not None and ball.y < y_min) or \
               (y_max is not None and ball.y > y_max):
                in_domain = False
        return in_domain

    def exert(self, dt, ball_list, **kwargs):
        raise NotImplementedError



class Linear(Physics):
    def __init__(self, dt, params = {}):
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        pass


class Uniform_Gravity(Physics):
    def __init__(self, dt, params = {}):
        assert "g" in params
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        for ball in ball_dict.values():
            if self.is_in_domain(ball):
                ball.vy += self.params["g"] * self.dt


class Brownian_Force(Physics):
    def __init__(self, dt, params = {}):
        assert "force_amp" in params
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        for ball in ball_dict.values():
            if self.is_in_domain(ball):
                ball.vx += np.random.randn() * self.params["force_amp"] * self.dt
                ball.vy += np.random.randn() * self.params["force_amp"] * self.dt


class Drag(Physics):
    def __init__(self, dt, params = {}):
        assert "coeff" in params and "power" in params
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        for ball in ball_dict.values():
            if self.is_in_domain(ball):
                v = np.sqrt(ball.vx ** 2 + ball.vy ** 2)
                ball.vx -= self.params["coeff"] * ball.vx / v * v ** self.params["power"] * self.dt
                ball.vy -= self.params["coeff"] * ball.vy / v * v ** self.params["power"] * self.dt


class Central_Force(Physics):
    def __init__(self, dt, params = {}):
        assert "coeff" in params and "power" in params and "center" in params
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        center_x, center_y = self.params["center"]
        force_coeff = self.params["coeff"]
        force_power = self.params["power"]
        for ball in ball_dict.values():
            if self.is_in_domain(ball):
                force_x = 0
                force_y = 0
                r = np.sqrt((ball.x - center_x) ** 2 + (ball.y - center_y) ** 2)
                force = force_coeff * r ** (force_power - 1)
                force_x += force * (center_x - ball.x)
                force_y += force * (center_y - ball.y)
                ball.vx += force_x * self.dt
                ball.vy += force_y * self.dt


class Point_Force_Paddle(Physics):
    def __init__(self, dt, params = {}):
        assert "coeff" in params and "power" in params
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        paddle_dict = kwargs["paddle_dict"]
        force_coeff = self.params["coeff"]
        force_power = self.params["power"]
        for ball in ball_dict.values():
            if self.is_in_domain(ball):
                force_x = 0
                force_y = 0
                for paddle in paddle_dict.values():
                    r = np.sqrt((ball.x - paddle.x) ** 2 + (ball.y - paddle.y) ** 2)
                    force = force_coeff * r ** (force_power - 1)
                    force_x += force * (paddle.x - ball.x)
                    force_y += force * (paddle.y - ball.y)
                ball.vx += force_x * self.dt
                ball.vy += force_y * self.dt


# Interaction forces:
class Pairwise_Force(Physics):
    def __init__(self, dt, params = {}):
        assert "coeff" in params and "power" in params
        self.dt = dt
        self.params = params

    def exert(self, ball_dict, **kwargs):
        force_coeff = self.params["coeff"]
        force_power = self.params["power"]
        for i, ball1 in enumerate(ball_dict.values()):
            if self.is_in_domain(ball1):
                force_x = 0
                force_y = 0
                for j, ball2 in enumerate(ball_dict.values()):
                    if j != i:
                        if self.is_in_domain(ball2):
                            r_x = ball2.x - ball1.x
                            r_y = ball2.y - ball1.y
                            r = np.sqrt(r_x ** 2 + r_y ** 2)
                            r = np.maximum(r, 1)
                            force = force_coeff * r ** (force_power - 1)
                            force_x += force * r_x
                            force_y += force * r_y
                ball1.vx += force_x * self.dt
                ball1.vy += force_y * self.dt
