import os


def replace_if_condition(condition, replace_value_if_true, replace_value_if_false=None):
    def f(x):
        if condition(x):
            return replace_value_if_true
        if replace_value_if_false is None:
            return x
        else:
            return replace_value_if_false
    return f


def add_suffix_to_files(dir_path, suffix):
    for file_name in set(os.listdir(dir_path)):
        os.rename(os.path.join(dir_path, file_name), os.path.join(dir_path, file_name+suffix))


def prepare_sample_dir(sample_dir_path):
    add_suffix_to_files(os.path.join(sample_dir_path, "csv"), ".csv")
    os.mkdir(os.path.join(sample_dir_path, "images"))


def get_kwargs(**kwargs):
    kwargs = dict(kwargs)
    for key, value in kwargs.copy().items():
        if value is None:
            kwargs.pop(key)
            continue
        try:
            iter(value)
        except TypeError:
            kwargs[key] = [value]
    return kwargs
