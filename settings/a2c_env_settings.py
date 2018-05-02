import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.util import new_dict

ENV_SETTINGS_CHOICE = {"envBounceBase": {
								 "num_balls": 1,
								 "no_bricks": True,
								 "ball_shape": "square",
								 "ball_radius": 2,
								 "ball_vmax": 4,
								 "num_paddles": 0,
								 "screen_height": 40,
						    	 "screen_width": 40,
						    	 "step_dt": 0.5,
						    	 "dt": 0.5,
								}
					  }
ENV_SETTINGS_CHOICE["envBounceStates"] = new_dict(ENV_SETTINGS_CHOICE["envBounceBase"],
										 {
										  "input_dims": (0, 1),
										 })

ENV_SETTINGS_CHOICE["envBounce1"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(4, 1), (15, 1), (29, 14), (20, 28), (1, 23)]],
										 })
ENV_SETTINGS_CHOICE["envBounce2"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(2, 2), (15, 4), (38, 10), (36, 38), (10, 35)]],
										 })
ENV_SETTINGS_CHOICE["envBounce3"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(3, 5), (38, 3), (37, 34), (16, 38), (3, 35)]],
										 })
ENV_SETTINGS_CHOICE["envBounce4"] = new_dict(ENV_SETTINGS_CHOICE["envBounceStates"],
										 {"boundaries": [[(3, 10), (18, 3), (36, 3), (38, 37), (4, 35)]],
										 })
