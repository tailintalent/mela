import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.util import new_dict

ENV_SETTINGS_CHOICE = {"envBounceBase": {
								 "num_balls": 1,
								 "no_bricks": True,
								 "ball_shape": "circle",
								 "ball_radius": 3,
								 "ball_vmax": 4,
								 "num_paddles": 0,
								 "screen_height": 39,
								 "screen_width": 39,
								 "max_range": (0, 39),
								 "step_dt": 1,
								 "dt": 0.1,
								}
					  }
ENV_SETTINGS_CHOICE["envBounceStates"] = new_dict(ENV_SETTINGS_CHOICE["envBounceBase"],
										 {
										  "input_dims": (0, 1),
										 })

ENV_SETTINGS_CHOICE["envBounce1"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(4, 3), (15, 3), (29, 14), (20, 28), (3, 23)]],
										 })
ENV_SETTINGS_CHOICE["envBounce2"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(3, 3), (15, 4), (37, 10), (36, 37), (10, 35)]],
										 })
ENV_SETTINGS_CHOICE["envBounce3"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(4, 7), (37, 3), (37, 34), (16, 37), (6, 34)]],
										 })
ENV_SETTINGS_CHOICE["envBounce4"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(3, 10), (18, 3), (35, 3), (38, 36), (8, 34)]],
										 })
