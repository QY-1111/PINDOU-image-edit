from .pindou_node import PindouMosaicPattern

NODE_CLASS_MAPPINGS = {
    "PindouMosaicPattern": PindouMosaicPattern,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PindouMosaicPattern": "拼豆马赛克图纸（带色号）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
