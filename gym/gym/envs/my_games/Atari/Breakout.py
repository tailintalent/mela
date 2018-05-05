import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from copy import deepcopy
from gym.envs.my_games.Atari.physics import get_physics


def get_item(item_source, k, num_items):
    if isinstance(item_source, list):
        if num_items > len(item_source):
            item = np.random.choice(item_source)
        else:
            item = item_source[k]
    else:
        item = item_source
    return item


class PhysicalObject(object):
    def __init__(self, x, y, vx, vy, shape = None):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.isdead = False
        self.grayscale = 255
        if shape is not None:
            self.height, self.width = shape
        else:
            self.height, self.width = 2, 2

    def step(self):
        raise NotImplementedError


    def get_inside_idx(self, screen_height, screen_width):
        if not self.isdead:
            x_range = range(max(0, np.ceil(self.x - self.width / float(2)).astype(int)), int(min(screen_width, np.floor(self.x + self.width / float(2) + 1))))
            y_range = range(max(0, np.ceil(self.y - self.height / float(2)).astype(int)), int(min(screen_height, np.floor(self.y + self.height / float(2) + 1))))
            idx = np.array(np.meshgrid(y_range, x_range)).transpose([2,1,0]).reshape(-1, 2)
            grayscale = self.grayscale * np.ones(len(idx))
            return idx, grayscale
        else:
            return None, None


    def isinside(self, x, y, offset = 0):
        if abs(self.x - x) <= self.width / float(2) + offset and abs(self.y - y) <= self.height / float(2) + offset:
            is_inside = True
            x_inside = self.width / float(2) + offset - abs(self.x - x)
            y_inside = self.height / float(2) + offset - abs(self.y - y)
            if x_inside > y_inside:
                if y < self.y:
                    col_direction = "y_small"
                else:
                    col_direction = "y_large"
            else:
                if x < self.x:
                    col_direction = "x_small"
                else:
                    col_direction = "x_large"
        else:
            is_inside = False
            col_direction = None
        return is_inside, col_direction



class Paddle(PhysicalObject):
    def __init__(self, x, y, vx = 0, vy = 0, shape = None, move_dx = 1, bounce_mode = "x_dependent"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.x_last = None
        self.y_last = None
        self.bounce_mode = bounce_mode
        self.isdead = False
        self.grayscale = 255
        if shape is not None:
            self.height, self.width = shape
        else:
            self.height, self.width = 2, 10
        self.move_dx = move_dx


    def step(self, action):
        self.x_last = deepcopy(self.x)
        self.y_last = deepcopy(self.y)
        if action == 2:
            self.x -= self.move_dx
        elif action == 3:
            self.x += self.move_dx


    def collide_with(self, object_dict):
        iscollide_list = []
        for Object in object_dict.values():
            iscollide, col_direction = self.isinside(Object.x, Object.y, offset = Object.radius)
            iscollide_list.append(iscollide)
            if iscollide:
                if col_direction[0] == "x":
                    Object.vx = -Object.vx
                    if col_direction == "x_small":
                        Object.x -= 2 * ((self.width / float(2) + Object.radius) - abs(Object.x - self.x))
                    else:
                        Object.x += 2 * ((self.width / float(2) + Object.radius) - abs(Object.x - self.x))
                elif col_direction[0] == "y":
                    Object.vy = - Object.vy
                    if self.bounce_mode == "plain":
                        pass
                    elif self.bounce_mode == "x_dependent":
                        Object.vx += (Object.x - self.x) / float(5) * 2
                    else:
                        raise Exception("bounce_mode {0} not recognized!".format(self.bounce_mode))
                    if col_direction == "y_small":
                        Object.y -= 2 * ((self.height / float(2) + Object.radius) - abs(Object.y - self.y))
                    else:
                        Object.y += 2 * ((self.height / float(2) + Object.radius) - abs(Object.y - self.y))
        reward = 0
        return np.any(iscollide_list), reward


class Brick(PhysicalObject):
    def __init__(self, x, y, vx = 0, vy = 0, shape = None, reward = 1, breakable = True):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        if shape is not None:
            self.height, self.width = shape
        else:
            self.height, self.width = 2, 4
        self.reward = reward
        self.breakable = breakable
        self.isdead = False
        self.grayscale = 255 if self.breakable else 150

    def step(self):
        pass


    def collide_with(self, object_dict):
        iscollide_list = []
        for Object in object_dict.values():
            iscollide, col_direction = self.isinside(Object.x, Object.y, offset = Object.radius)
            iscollide_list.append(iscollide)
            if iscollide:
                if col_direction[0] == "x":
                    Object.vx = -Object.vx
                    if col_direction == "x_small":
                        Object.x -= 2 * ((self.width / float(2) + Object.radius) - abs(Object.x - self.x))
                    else:
                        Object.x += 2 * ((self.width / float(2) + Object.radius) - abs(Object.x - self.x))
                elif col_direction[0] == "y":
                    Object.vy = -Object.vy
                    if col_direction == "y_small":
                        Object.y -= 2 * ((self.height / float(2) + Object.radius) - abs(Object.y - self.y))
                    else:
                        Object.y += 2 * ((self.height / float(2) + Object.radius) - abs(Object.y - self.y))
                if self.breakable:
                    self.isdead = True
        if np.any(iscollide_list):
            reward = self.reward
        else:
            reward = 0
        return np.any(iscollide_list), reward


class Ball(PhysicalObject):
    def __init__(self, x, y, vx, vy, radius = 1, shape = "square"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.x_last = None
        self.y_last = None
        self.radius = radius
        self.shape = shape
        self.width = self.height = self.radius * 2
        self.isdead = False
        self.grayscale = 255


    def step(self, dt, **kwargs):
        self.x_last = deepcopy(self.x)
        self.y_last = deepcopy(self.y)
        self.x += self.vx * dt
        self.y += self.vy * dt


    def get_inside_idx(self, screen_height, screen_width):
        if not self.isdead:
            if self.shape == "square":
                x_range = range(max(0, np.ceil(self.x - self.radius).astype(int)), int(min(screen_width, np.floor(self.x + self.radius + 1))))
                y_range = range(max(0, np.ceil(self.y - self.radius).astype(int)), int(min(screen_height, np.floor(self.y + self.radius + 1))))
                idx = np.array(np.meshgrid(y_range, x_range)).transpose([2,1,0]).reshape(-1, 2)
            elif self.shape == "circle":
                x_range = range(max(0, np.ceil(self.x - self.radius).astype(int)), int(min(screen_width, np.floor(self.x + self.radius + 1))))
                y_range = range(max(0, np.ceil(self.y - self.radius).astype(int)), int(min(screen_height, np.floor(self.y + self.radius + 1))))
                idx = np.array(np.meshgrid(y_range, x_range)).transpose([2,1,0]).reshape(-1, 2)
                center = np.array([np.floor(self.y) + 0.5, np.floor(self.x) + 0.5]) 
                distance = ((idx - center) ** 2).sum(1)
                idx = idx[distance <= max(self.radius ** 2 - 1, 1)]
            elif self.shape == "square_hollow":
                x_min = max(0, np.ceil(self.x - self.radius).astype(int))
                x_max = int(min(screen_width, np.floor(self.x + self.radius)))
                y_min = max(0, np.ceil(self.y - self.radius).astype(int))
                y_max = int(min(screen_height, np.floor(self.y + self.radius)))
                x_range = np.array([x_min, x_max])
                y_range = np.array([y_min, y_max])
                x_range_all = range(x_min, x_max + 1)
                y_range_all = range(y_min, y_max + 1)
                idx_v = np.array(np.meshgrid(y_range, x_range_all)).transpose([2,1,0]).reshape(-1, 2)
                idx_h = np.array(np.meshgrid(y_range_all, x_range)).transpose([2,1,0]).reshape(-1, 2)
                idx = np.concatenate([idx_h, idx_v])
            elif self.shape == "plus":
                center_x = np.round(self.x)
                center_y = np.round(self.y)
                horizontal = range(int(max(0, center_x - self.radius + 1)), int(min(screen_width, center_x + self.radius)))
                vertical = range(int(max(0, center_y - self.radius + 1)), int(min(screen_height, center_y + self.radius)))
                idx_h = np.stack([np.ones(len(horizontal)) * center_y, horizontal], 1)
                idx_v = np.stack([vertical, np.ones(len(vertical)) * center_x], 1)
                idx = np.concatenate([idx_h, idx_v])
            else:
                raise Exception("ball's shape {0} not recognized!".format(self.shape))
            grayscale = self.grayscale * np.ones(len(idx))
            return idx, grayscale
        else:
            return None, None



def check_cross(boundary_line, trajectory_line, v):
    ((u1, v1), (u2, v2)), ((x1, y1), (x2, y2)) = boundary_line, trajectory_line
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = x1 * y2 - x2 * y1
    A2 = v2 - v1
    B2 = u1 - u2
    C2 = u1 * v2 - u2 * v1
    det = A1 * B2 - A2 * B1
    if det == 0:
        is_cross = False
    else:
        x = (B2 * C1 - B1 * C2) / float(det)
        y = (A1 * C2 - A2 * C1) / float(det)
        if min(x1, x2) <= x <= max(x1, x2) and \
           min(y1, y2) <= y <= max(y1, y2):
            is_cross = True
        else:
            is_cross = False

    if is_cross:
        # Get reflected point:
        coeff = -2 * (A2 * x2 + B2 * y2 - C2) / float(A2 ** 2 + B2 ** 2)
        x_refl = A2 * coeff + x2
        y_refl = B2 * coeff + y2

        # Get reflected velocity:
        vx, vy = v
        nx = A2 / np.sqrt(A2 ** 2 + B2 ** 2)
        ny = B2 / np.sqrt(A2 ** 2 + B2 ** 2)
        v_dot = vx * nx + vy * ny
        vx_refl = vx - 2 * v_dot * nx
        vy_refl = vy - 2 * v_dot * ny

        return is_cross, (x_refl, y_refl), (vx_refl, vy_refl)
    else:
        return is_cross, None, None


class Breakout_Custom(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, env_settings = {}):
        # The following parameters can be set by passing in env_settings in gym.make(env, env_settings):
        self.height = env_settings["screen_height"] if "screen_height" in env_settings else 84
        self.width = env_settings["screen_width"] if "screen_width" in env_settings else 84
        self.state_repr = env_settings["state_repr"] if "state_repr" in env_settings else "image"
        self.info_contents = env_settings["info_contents"] if "info_contents" in env_settings else []
        self.step_dt = env_settings["step_dt"] if "step_dt" in env_settings else 1
        self.dt = env_settings["dt"] if "dt" in env_settings else 0.25
        self.physics_list = get_physics(env_settings["physics"] if "physics" in env_settings else [], dt = self.dt)
        self.boundaries = env_settings["boundaries"] if "boundaries" in env_settings else "default"

        self.brick_y_range = env_settings["brick_y_range"] if "brick_y_range" in env_settings else (self.height * 0.27, self.height * 0.44)
        self.middle_unbreakable_bricks = env_settings["middle_unbreakable_bricks"] if "middle_unbreakable_bricks" in env_settings else False
        self.no_bricks = env_settings["no_bricks"] if "no_bricks" in env_settings else False
        if self.no_bricks:
            self.brick_y_range = (0, self.height * 0.25)

        self.num_paddles = env_settings["num_paddles"] if "num_paddles" in env_settings else 1
        self.paddle_y = env_settings["paddle_y"] if "paddle_y" in env_settings else int(self.height * 0.9)
        self.paddle_width = env_settings["paddle_width"] if "paddle_width" in env_settings else 10
        self.paddle_move_dx = env_settings["paddle_move_dx"] if "paddle_move_dx" in env_settings else 5
        self.paddle_bounce_mode = env_settings["paddle_bounce_mode"] if "paddle_bounce_mode" in env_settings else "x_dependent"

        self.num_balls = env_settings["num_balls"] if "num_balls" in env_settings else 1
        self.ball_vmax = env_settings["ball_vmax"] if "ball_vmax" in env_settings else 3
        self.ball_radius = env_settings["ball_radius"] if "ball_radius" in env_settings else 1
        self.ball_shape = env_settings["ball_shape"] if "ball_shape" in env_settings else "square"

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (1, self.height, self.width))
        self.viewer = None
        self.lives = 6

        self._seed()
        self.ball_dict = {}
        self._reset(start_game = True)


    def _reset_bricks_paddle(self):
        brick_height = (self.brick_y_range[1] - self.brick_y_range[0]) / float(6)
        brick_width = self.width / float(18)
        self.brick_dict = {}
        reward_dict = {0:7, 1: 7, 2: 4, 3: 4, 4:1, 5: 1}
        start = int(self.brick_y_range[0] / brick_height)
        if not self.no_bricks:
            k = 0
            for i in range(start, int(self.brick_y_range[1] / brick_height)):
                for j in range(int(self.width / brick_width)):
                    if self.middle_unbreakable_bricks and i == int(self.brick_y_range[1] / brick_height) - 1 \
                        and j >= int(0.25 * self.width / brick_width) and j <= int(0.75 * self.width / brick_width):
                        brick_breakable = False
                        reward_middlewall = self.middle_unbreakable_bricks["reward"] if isinstance(self.middle_unbreakable_bricks, dict) and \
                                        "reward" in self.middle_unbreakable_bricks else 0
                    else:
                        brick_breakable = True
                    brick = Brick(x = brick_width * j + brick_width / float(2),
                                  y = brick_height * i + brick_height / float(2),
                                  shape = (brick_height, brick_width),
                                  reward = reward_dict[i - start] if brick_breakable else reward_middlewall,
                                  breakable = brick_breakable,
                                 )
                    self.brick_dict[k] = brick
                    k += 1

        # Initialize paddle:
        self.paddle_dict = {}
        self.paddle_y_list = []
        for i in range(self.num_paddles):
            paddle_height = 1.5
            paddle_x = int(np.random.rand() * (self.width - 0.4 * self.width) + 0.2 * self.width)
            paddle_y = get_item(self.paddle_y, i, self.num_paddles)
            paddle_width = get_item(self.paddle_width, i, self.num_paddles)
            paddle_move_dx = get_item(self.paddle_move_dx, i, self.num_paddles)
            paddle_bounce_mode = get_item(self.paddle_bounce_mode, i, self.num_paddles)
            self.paddle_y_list.append(paddle_y)

            paddle = Paddle(x = paddle_x,
                            y = paddle_y,
                            shape = (paddle_height, paddle_width),
                            move_dx = paddle_move_dx,
                            bounce_mode = paddle_bounce_mode,
                           )
            self.paddle_dict[i] = paddle


    def _reset_ball(self):
        # Initialize ball:
        self.ball_dict = {}
        min_paddle_y = np.min(self.paddle_y_list) if len(self.paddle_y_list) > 0 else self.height

        for i in range(self.num_balls):
            # Ball shape and radius:
            ball_shape = get_item(self.ball_shape, i, self.num_balls)
            ball_radius = get_item(self.ball_radius, i, self.num_balls)

            # Ball position and velocity:
            ball_x = np.random.rand() * (self.width - 0.4 * self.width) + 0.2 * self.width
            ball_y = np.random.rand() * (min_paddle_y - 0.2 * self.height - 2 * ball_radius - self.brick_y_range[1]) + self.brick_y_range[1]
            if self.num_paddles > 0:
                if np.random.randint(2) == 1:
                    ball_angle = np.random.rand() * np.pi / 5 + np.pi / 4
                else:
                    ball_angle = np.pi * 3 / 4 - np.random.rand() * np.pi / 5
                if self.no_bricks:
                    ball_angle += np.pi
            else:
                ball_angle = np.random.rand() * np.pi * 2
            ball_vx = self.ball_vmax * (0.8 + 0.2 * np.random.rand()) * np.cos(ball_angle)
            ball_vy = self.ball_vmax * (0.8 + 0.2 * np.random.rand()) * np.sin(ball_angle)

            # Make ball:
            ball = Ball(x = ball_x, y = ball_y, vx = ball_vx, vy = ball_vy, radius = ball_radius, shape = ball_shape)
            self.ball_dict[i] = ball


    def _reset(self, start_game = False):
        self.game_started = False
        self.lives -= 1
        if not start_game and len(self.brick_dict) == 0:
            self.lives = 0
        if self.lives <= 0 or start_game:
            self.lives = 5
            self._reset_bricks_paddle()
        return self.get_observation()


    def _seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_observation(self):
        obs = np.zeros((self.height, self.width))
        object_region = []
        object_grayscale = []
        for paddle in self.paddle_dict.values():
            paddle_region, paddle_grayscale = paddle.get_inside_idx(self.height, self.width)
            object_region.append(paddle_region)
            object_grayscale.append(paddle_grayscale)
        for brick in self.brick_dict.values():
            brick_region, brick_grayscale = brick.get_inside_idx(self.height, self.width)
            object_region.append(brick_region)
            object_grayscale.append(brick_grayscale)
        for ball in self.ball_dict.values():
            ball_region, ball_grayscale = ball.get_inside_idx(self.height, self.width)
            object_region.append(ball_region)
            object_grayscale.append(ball_grayscale)
        if len(object_region) >= 1:
            object_region = np.concatenate(object_region, 0).astype(int)
            obs[object_region[:,0], object_region[:, 1]] = np.concatenate(object_grayscale, 0).astype(int)
        return np.expand_dims(obs, 0)


    def get_state(self, include_bricks = False):
        state = {"paddle": {}, "ball": {}}
        for i, paddle in self.paddle_dict.items():
            state["paddle"][i] = paddle.x
        for i, ball in self.ball_dict.items():
            state["ball"][i] = [ball.x, ball.y]
        if include_bricks:
            state["bricks"] = {}
            for i, brick in self.brick_dict.items():
                state["bricks"][i] = [brick.x, brick.y]
        return state



    def collide_with_edge(self, object_dict):
        bouncing_info = {}
        for name, Object in object_dict.items():
            bouncing_info[name] = []
            if self.boundaries == "default":
                if Object.x <= Object.width / float(2):
                    Object.x = Object.width - Object.x
                    Object.vx = - Object.vx
                    bouncing_info[name].append(3)
                elif Object.x >= self.width - Object.width / float(2):
                    Object.x = 2 * (self.width - Object.width / float(2)) - Object.x
                    Object.vx = - Object.vx
                    bouncing_info[name].append(4)

                if Object.y <= Object.height / float(2):
                    Object.y = Object.height - Object.y
                    Object.vy = - Object.vy
                    bouncing_info[name].append(1)
                elif Object.y >= self.height - Object.height / float(2):
                    if len(self.paddle_dict) > 0:
                        Object.isdead = True
                    else:
                        Object.y = 2 * (self.height - Object.height / float(2)) - Object.y
                        Object.vy = - Object.vy
                        bouncing_info[name].append(2)
            else:
                for i in range(len(self.boundaries)):
                    u1, v1 = self.boundaries[i]
                    if i == len(self.boundaries) - 1:
                        u2, v2 = self.boundaries[0]
                    else:
                        u2, v2 = self.boundaries[i + 1]
                    x1, x2 = Object.x_last, Object.x
                    y1, y2 = Object.y_last, Object.y

                    # Check if there is collision:
                    is_cross, point_refl, v_refl = check_cross(boundary_line = ((u1, v1), (u2, v2)), trajectory_line = ((x1, y1), (x2, y2)), v = (Object.vx, Object.vy))
                    if is_cross:
                        Object.x, Object.y = point_refl
                        Object.vx, Object.vy = v_refl
                        bouncing_info[name].append(i + 1)
            if len(bouncing_info[name]) == 0:
                bouncing_info[name].append(0)
            bouncing_info[name] == sorted(bouncing_info[name])
        return bouncing_info


    def check_collision(self):
        for paddle in self.paddle_dict.values():
            paddle.collide_with(self.ball_dict)
        ball_bouncing_info = self.collide_with_edge(self.ball_dict)
        paddle_bouncing_info = self.collide_with_edge(self.paddle_dict)
        reward_list = []
        for brick in self.brick_dict.values():
            iscollide, reward = brick.collide_with(self.ball_dict)
            reward_list.append(reward)
        return np.sum(reward_list), ball_bouncing_info, paddle_bouncing_info


    def check_dead(self):
        brick_dead_list = []
        for i, brick in self.brick_dict.items():
            if brick.isdead:
                brick_dead_list.append(i)
        for i in reversed(brick_dead_list):
            self.brick_dict.pop(i)

        ball_dead_list = []
        for i, ball in self.ball_dict.items():
            if ball.isdead:
                ball_dead_list.append(i)
        for i in reversed(ball_dead_list):
            self.ball_dict.pop(i)

        return np.sum(ball_dead_list)



    def _step(self, action):
        reward_list = []
        num_dead_balls_list = []
        # Check whether to start the game:
        if not self.game_started and np.any(action == 1):
            self._reset_ball()
            self.game_started = True

        for i in range(int(self.step_dt / self.dt)):
            # Physics:
            for physics in self.physics_list:
                physics.exert(self.ball_dict, paddle_dict = self.paddle_dict)
            for ball in self.ball_dict.values():
                ball.step(self.dt)

            # Paddle operation on the ball:
            if i == 0:
                for k, paddle in self.paddle_dict.items():
                    action_k = action[k] if len(self.paddle_dict) > 1 else action
                    paddle.step(action_k)

            # Check collision of the ball with bricks:
            reward, ball_bouncing_info, paddle_bouncing_info = self.check_collision()
            reward_list.append(reward)
            num_dead_balls = self.check_dead()
            num_dead_balls_list.append(num_dead_balls)

        # Obtain states and reward for this frame:
        if self.state_repr == "image":
            self.state = self.get_observation()
        elif self.state_repr == "coordinates":
            self.state = self.get_state(include_bricks = False)
        else:
            raise Exception("state_repr {0} not recognized!".format(self.state_repr))
        total_reward = np.sum(reward_list)
        if self.num_balls > 1:
            total_reward -= np.sum(num_dead_balls_list)

        # Determine whether the episode ends:
        if self.game_started and (len(self.ball_dict) == 0 or (not self.no_bricks and len(self.brick_dict) == 0)):
            done = True
        else:
            done = False
        info = {}
        info["ball_bouncing_info"] = ball_bouncing_info
        info["paddle_bouncing_info"] = paddle_bouncing_info

        # Obtain other information (not used by the agent):
        for info_content in self.info_contents:
            if info_content == "coordinates":
                info[info_content] = self.get_state(include_bricks = True)
            else:
                raise Exception("info_content {0} not recognized!".format(info_content))
        return self.state, total_reward, done, info


    def _render(self, mode='human', close = False):
        if self.viewer is None:
            self.viewer = Viewer(height = self.height, width = self.width, resize_ratio = 3)
        if not hasattr(self, "state"):
            self.state = self.get_observation()
        if self.state_repr == "image":
            self.viewer.render(self.state.squeeze(0))
        elif self.state_repr == "coordinates":
            self.viewer.render(self.get_observation().squeeze(0))
        else:
            raise Exception("self.state_repr {0} not recognized!".format(self.state_repr))


class Viewer(object):
    def __init__(self, height = 84, width = 84, resize_ratio = 1):
        from pyglet.gl import glEnable, glBlendFunc
        from pyglet.gl import GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        from pyglet import window, image
        from pygarrayimage.arrayimage import ArrayInterfaceImage
        self.width = width
        self.height = height
        self.resize_ratio = resize_ratio
        self.window = window.Window(width = int(width * resize_ratio), height = int(height * resize_ratio), visible = False, resizable=True)
        self.arr = np.zeros((int(height * resize_ratio), int(width * resize_ratio)), dtype = np.uint8)
        self.aii = ArrayInterfaceImage(self.arr)
        self.img = self.aii.texture

        checks = image.create(32, 32, image.CheckerImagePattern())
        self.background = image.TileableTexture.create_for_image(checks)
        self.window.set_visible()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


    def render(self, new_arr):
        assert new_arr.shape == (self.height, self.width), "the array shape {0} should be consistent with stipulated shape {1}!".format(new_arr.shape, (self.height, self.width))
        if self.resize_ratio != 1:
            new_arr = np.repeat(np.repeat(new_arr, int(self.resize_ratio), 0), int(self.resize_ratio), 1)
        self.window.dispatch_events()
        self.background.blit_tiled(0, 0, 0, self.window.width, self.window.height)
        self.img.blit(0, 0, 0)
        self.window.flip()

        np.place(self.arr, self.arr >= -1, new_arr.astype('uint8')[::-1,:])
        self.aii.dirty() # dirty the ArrayInterfaceImage because the data changed

    def close(self):
        self.window.close()