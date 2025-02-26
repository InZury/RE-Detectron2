from collections import namedtuple


class ShapeSpec(namedtuple("ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        param = (channels, height, width, stride)
        return super().__new__(cls, *param)
