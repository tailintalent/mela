import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.util import new_dict

ENV_SETTINGS_CHOICE = {"env1.1": {
								 "num_balls": 1,
								 "no_bricks": True,
								 "ball_shape": "square",
								 "ball_radius": 1,
								 "ball_vmax": 4,
								 "num_paddles": 0,
								 "screen_height": 30,
						    	 "screen_width": 30,
								}
					  }


# Testing different bouncing geometries:
ENV_SETTINGS_CHOICE["env6.0"] = new_dict(ENV_SETTINGS_CHOICE["env1.1"],
										 {"boundaries": [[(1,1), (1,29), (29, 29), (29, 1)]],
										  "num_balls": 1,
										  "input_dims": (0,1),
										 })

ENV_SETTINGS_CHOICE["env6.1"] = new_dict(ENV_SETTINGS_CHOICE["env6.0"],
										 {"boundaries": [[(4, 1), (15, 1), (29, 14), (20, 28), (1, 23)]],
										 })
ENV_SETTINGS_CHOICE["env6.2"] = new_dict(ENV_SETTINGS_CHOICE["env6.0"],
										 {"boundaries": [[(1, 1), (15, 4), (28, 10), (26, 28), (10, 25)]],
										 })
ENV_SETTINGS_CHOICE["env6.3"] = new_dict(ENV_SETTINGS_CHOICE["env6.0"],
										 {"boundaries": [[(2, 5), (28, 1), (27, 24), (16, 28), (2, 25)]],
										 })
ENV_SETTINGS_CHOICE["env6.4"] = new_dict(ENV_SETTINGS_CHOICE["env6.0"],
										 {"boundaries": [[(3, 10), (18, 1), (26, 2), (29, 27), (3, 25)]],
										 })
