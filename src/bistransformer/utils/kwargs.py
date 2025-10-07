from typing import Any, Dict

def apply_kwargs(defaults: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Allow only known keys to be overridden. 
    Unknown keys raise an error to prevent silent bugs.
    """
    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise TypeError(f"Unknown kwargs: {unknown}. Allowed: {list(defaults.keys())}")
    conf = defaults.copy()
    conf.update(kwargs)
    return conf
