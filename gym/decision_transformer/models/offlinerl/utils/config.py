from collections import OrderedDict


del_attr = ["function", "module"]

def parse_config(cfg_module):
    args = [ i for i in dir(cfg_module) if not i.startswith("__")]

    config = dict()
    for arg in args:
        k = arg
        v = getattr(cfg_module, arg)
        if type(v).__name__ in del_attr and k != "device":
            continue
        else:
            config[k] = v

    
    return config
    