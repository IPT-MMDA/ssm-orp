from .blocks import SimpleSSMBlock
from .baseline import StandardSSM
from .shared import SharedSSM

def get_parameter_count(model) -> int:
    """Рахує кількість параметрів, які навчаються."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

__all__ = [
    "SimpleSSMBlock",
    "StandardSSM",
    "SharedSSM",
    "get_parameter_count"
]