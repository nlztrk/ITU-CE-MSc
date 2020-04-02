def flatten_unflatten(f):
    def wrapper(*args, **kwargs):
        obj, x = args
        x_t = x.transpose((0, 2, 3, 1))
        x_flat = x_t.reshape(-1, x.shape[1])
        out = f(obj, x_flat, **kwargs)
        out_reshaped = out.reshape(*x_t.shape)
        out = out_reshaped.transpose((0, 3, 1, 2))
        return out
    return wrapper