import numpy as np

alg_colors = {
    "unconstrained": "#AA5D1F",
    "LR": "#BA2DC1",
    "RSPO": "#6C2896",
    "SQRL": "#D43827",
    "RP": "#4899C5",
    "RCPO": "#34539C",
    "RRL_MF": "#60CC38",
    "RRL_MB": "#349C26"
}

alg_names = {
    "unconstrained": "Unconstrained",
    "LR": "LR",
    "RSPO": "RSPO",
    "SQRL": "SQRL",
    "RP": "RP",
    "RCPO": "RCPO",
    "RRL_MF": "Ours: Recovery RL (MF Recovery)",
    "RRL_MB": "Ours: Recovery RL (MB Recovery)"
}


def get_color(algname, alt_color_map={}):
    if algname in alg_colors:
        return alg_colors[algname]
    elif algname in alt_color_map:
        return alt_color_map[algname]
    else:
        return np.random.rand(3, )


def get_legend_name(algname, alt_name_map={}):
    if algname in alg_names:
        return alg_names[algname]
    elif algname in alt_name_map:
        return alt_name_map[algname]
    else:
        return algname
