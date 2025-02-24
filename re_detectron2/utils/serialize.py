import cloudpickle


class PicklableWrapper(object):
    def __init__(self, obj):
        self.obj = obj

    def __reduce__(self):
        return cloudpickle.loads, (cloudpickle.dumps(self.obj), )

    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)

    def __getattr__(self, attr):
        if attr not in ["obj"]:  # TODO Check name
            return getattr(self.obj, attr)

        return getattr(self, attr)
