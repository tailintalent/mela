from copy import deepcopy
# The dictionary of weight_type, bias_type and activation for struct_string:
# The ones with "-int" means that the elements are integers and non-trainable.
AVAILABLE_REG = ["L1", "L2", "layer_L1", "S_entropy", "S_entropy_activation", "param"]
Default_Activation = "linear"
COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]