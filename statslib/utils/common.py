import IPython as _ipython
import functools as _functools
import operator as _operator


from functools import wraps
import datetime
from collections import namedtuple

from statslib.utils.log import Logger
from statslib.utils.dates import return_time_delta_list

flatten_lst = lambda my_lst: _functools.reduce(_operator.iconcat, my_lst, [])


def print_list_columnwise(ths_list, ths_title=None, sort=False, *args,
                          **kwargs):
    if ths_title is not None:
        print("\n{}".format(ths_title))
    print("-" * 100)
    if len(ths_list) > 50:
        my_displaywidth = 100
    else:
        my_displaywidth = 40

    ths_list = [str(val) for val in ths_list]
    if sort:
        ths_list = sorted(ths_list)
    print(
        _ipython.utils.text.columnize(
            ths_list,
            displaywidth=kwargs.get('displaywidth', my_displaywidth),
            spread=kwargs.get('spread', False),
            row_first=kwargs.get('row_first', False),
            separator=kwargs.get('separator', '| '),
        ))


def print_attributes(ths_obj,
                     public_only=False,
                     return_attributes=False,
                     silent=False,
                     methods=False,
                     members=False,
                     *args,
                     **kwargs):
    ths_attributes_list = dir(ths_obj)
    if methods:
        ths_attributes_list = [attr for attr in ths_attributes_list if inspect.ismethod(getattr(ths_obj, attr))]

    if members:
        ths_attributes_list = [attr for attr in ths_attributes_list if not inspect.ismethod(getattr(ths_obj, attr))]

    ths_dbl_uscore_attributes = sorted([
        attribute for attribute in ths_attributes_list
        if attribute.startswith('_') and attribute[1] == '_'
    ])
    ths_uscore_attributes = [
        attribute for attribute in ths_attributes_list
        if attribute.startswith('_')
    ]
    ths_uscore_attributes = set(ths_uscore_attributes).difference(
        set(ths_dbl_uscore_attributes))
    ths_uscore_attributes = sorted(list(ths_uscore_attributes))

    ths_attributes = set(ths_attributes_list).difference(
        set(ths_uscore_attributes).union(set(ths_dbl_uscore_attributes)))
    ths_attributes = sorted(list(ths_attributes))

    if not silent:
        if public_only:
            print_list_columnwise(
                sorted(ths_attributes), "Public attributes", *args, **kwargs)
        else:
            print_list_columnwise(
                sorted(ths_attributes), "Public attributes", *args, **kwargs)
            print_list_columnwise(
                sorted(ths_uscore_attributes), "Private attributes:", *args,
                **kwargs)
            print_list_columnwise(
                sorted(ths_dbl_uscore_attributes), "Class specific attributes:",
                *args, **kwargs)

    if return_attributes:
        return {
            'public_atr': ths_attributes,
            'private_atr': ths_uscore_attributes,
            'class_atr': ths_dbl_uscore_attributes
        }


def align_two_lists(orig_l, target_l):
    target_l[:] = [
        target_l[i]
        for i in [target_l.index(val) for val in orig_l if val in target_l]
    ]


def report_run_time(long_format=True, f_name=None):

    def next_decorator(a_func):

        @wraps(a_func)
        def decorated(*args, **kwargs):
            started = datetime.datetime.now()
            result = a_func(*args, **kwargs)
            finished = datetime.datetime.now()
            time_delta = return_time_delta_list(started, finished)
            if long_format:
                Logger().get_logger().info(
                    'Function {} took {:.2f} hours {:.2f} minutes and {:.0f} seconds.'
                    .format(a_func.__name__ if f_name is None else f_name,
                            time_delta[0], time_delta[1], time_delta[2]))
            else:
                Logger().get_logger().info(
                    'Function {} took {:.2f} minutes and {:.0f} seconds.'.
                    format(a_func.__name__ if f_name is None else f_name,
                           time_delta[1], time_delta[2]))
            return result

        return decorated

    return next_decorator


def to_namedtuple(d, recursive=True):
    if isinstance(d, dict):
        d = d.copy()
        if recursive:
            for k, v in d.items():
                d[k] = to_namedtuple(v, recursive)
        d = namedtuple('_', d.keys())(**d)

    return d


def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])


def powerset(iterable, skip_empty_set=False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from copy import deepcopy
    from itertools import chain, combinations
    s = list(iterable)
    if skip_empty_set:
        rng = range(1,len(s)+1)
    else:
        rng = range(0, len(s) + 1)
    return chain.from_iterable(combinations(s, r) for r in rng)