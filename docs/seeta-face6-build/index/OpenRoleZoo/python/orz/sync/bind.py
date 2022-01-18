#!/usr/bin/env python
# coding: utf-8

from copy import copy


class Placeholder(object):
    def __init__(self, index_or_key):
        self.index = None
        self.key = None
        if isinstance(index_or_key, int):
            self.index = index_or_key
        elif isinstance(index_or_key, str):
            self.key = index_or_key
        elif isinstance(index_or_key, unicode):
            self.key = index_or_key
        else:
            raise RuntimeError('Can not hold place with %s of %s' % (type(index_or_key), index_or_key))

    def __str__(self):
        return 'Placeholder(%s)' % repr(self.index if self.index is not None else self.key)

    def __repr__(self):
        return self.__str__()


def set_list(l, i, val):
    while len(l) <= i:
        l.append(None)
    old_val = l[i]
    l[i] = val
    return old_val


def set_dict(d, key, val):
    old_val = None
    if key in d:
        old_val = d[key]
    d[key] = val
    return old_val


class Bound(object):
    def __init__(self, func, *args, **kwargs):
        self.__func = func
        self.__args = list(args)
        self.__kwargs = dict(kwargs)

        self.__index_to_args = []   # index: index or key
        self.__key_to_args = {}     # key: index or key

        self.__extra_keys = []     # extra keys

        for i in range(len(self.__args)):
            arg = self.__args[i]
            if isinstance(arg, Placeholder):
                if arg.index is not None:
                    if len(self.__key_to_args) > 0:
                        raise TypeError('key place holder must after index holder')
                    if set_list(self.__index_to_args, arg.index, i) is not None:
                        raise TypeError('duplicate index')
                else:
                    if set_dict(self.__key_to_args, arg.key, i) is not None:
                        raise TypeError('duplicate key')
                    self.__extra_keys.append(arg.key)

        for key in self.__kwargs.keys():
            arg = self.__kwargs[key]
            if isinstance(arg, Placeholder):
                if arg.index is not None:
                    if set_list(self.__index_to_args, arg.index, key) is not None:
                        raise TypeError('duplicate index')
                else:
                    if set_dict(self.__key_to_args, arg.key, key) is not None:
                        raise TypeError('duplicate key')

        # check index place holder
        for arg in self.__index_to_args:
            if arg is None:
                raise TypeError('place holders not all defined')

    def __call__(self, *args, **kwargs):
        unknown_keys = set(kwargs.keys()) - set(self.__key_to_args)
        if len(unknown_keys) > 0:
            raise TypeError('got an unexpected keyword argument %s' % (str(unknown_keys)[5:-2]))

        given_args_num = len(args) + len(kwargs)
        need_args_num = len(self.__index_to_args) + len(self.__key_to_args)

        if given_args_num != need_args_num:
            raise TypeError('takes exactly %d arguments (%d given)' % (need_args_num, given_args_num))

        local_args = copy(self.__args)
        local_kwargs = copy(self.__kwargs)

        argi = 0
        while argi < len(self.__index_to_args):
            arg = self.__index_to_args[argi]
            if isinstance(arg, int):
                if not isinstance(set_list(local_args, arg, args[argi]), Placeholder):
                    raise TypeError('duplicate index')
            else:
                if not isinstance(set_dict(local_kwargs, arg, args[argi]), Placeholder):
                    raise TypeError('duplicate key')
            argi += 1

        while argi < len(args):
            argkeyi = argi - len(self.__index_to_args)
            key = self.__extra_keys[argkeyi]
            arg = self.__key_to_args[key]
            if not isinstance(set_list(local_args, arg, args[argi]), Placeholder):
                raise TypeError('duplicate index')
            argi += 1

        for key in kwargs.keys():
            arg = self.__key_to_args[key]
            if isinstance(arg, int):
                if not isinstance(set_list(local_args, arg, kwargs[key]), Placeholder):
                    raise TypeError('duplicate index')
            else:
                if not isinstance(set_dict(local_kwargs, arg, kwargs[key]), Placeholder):
                    raise TypeError('duplicate key')

        self.__func(*local_args, **local_kwargs)

    def __str__(self):
        first = True
        fmt = ['Bound(']
        for arg in self.__args:
            if len(fmt) > 1:
                fmt.append(', ')
            fmt.append(repr(arg))
        for key in self.__kwargs:
            arg = self.__kwargs[key]
            if len(fmt) > 1:
                fmt.append(', ')
            fmt.append('%s=%s' % (str(key), repr(arg)))
        fmt.append(')')
        fmt = ''.join(fmt)
        return fmt

    def __repr__(self):
        return self.__str__()


def print_args(*args, **kwargs):
    print 'args:', args
    print 'kwargs:', kwargs


def bind(func, *args, **kwargs):
    return Bound(func, *args, **kwargs)


if __name__ == '__main__':

    f = bind(print_args, Placeholder(1), "", Placeholder('t'), a=1, b=Placeholder(0))

    print f

    f(1, 2, 3)