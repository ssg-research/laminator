#!/usr/bin/env python

import hashlib
import io

class MeasuredBytesIO(io.BytesIO):
    def __init__(self, initial, hasher, *args, **kwargs):
        self.hasher = hasher
        self.hasher.update(initial)
        super().__init__(initial, *args, **kwargs)

class MeasuredStringIO(io.StringIO):
    def __init__(self, initial, hasher, *args, **kwargs):
        self.hasher = hasher
        self.hasher.update(bytes(initial, 'utf8'))
        super().__init__(initial, *args, **kwargs)

def open_measured(path, mode, hasher=None):
    if not hasher:
        hasher = hashlib.sha512()

    if 'w' in mode or 'a' in mode or 'x' in mode or '+' in mode:
        raise IOError("Measured open can't be writable")

    if 'b' in mode:
        with open(path, mode) as fh:
            data = fh.read()
            return MeasuredBytesIO(data, hasher)
    else:
        with open(path, mode) as fh:
            data = fh.read()
            return MeasuredStringIO(data, hasher)

class MeasuredBytesIOWrite(io.BytesIO):
    def __init__(self, hasher, *args, **kwargs):
        self.hasher = hasher
        super().__init__(*args, **kwargs)
    
    def write(self, b):
        self.hasher.update(b)
        return super().write(b)

class MeasuredStringIOWrite(io.StringIO):
    def __init__(self, hasher, *args, **kwargs):
        self.hasher = hasher
        super().__init__(*args, **kwargs)
    
    def write(self, s):
        self.hasher.update(s.encode('utf-8'))
        return super().write(s)

def open_measured_write(path, mode, hasher=None):
    if not hasher:
        hasher = hashlib.sha512()
    
    # Check for valid write modes
    if 'r' in mode:
        raise IOError("Measured open must be writable")

    if 'b' in mode:
        return MeasuredBytesIOWrite(hasher), open(path, mode)
    else:
        return MeasuredStringIOWrite(hasher), open(path, mode)