import logging
from time import time
import math
from random import randint, random
import itertools
import unittest

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
    __global__ void newtons_law(float *f_out, float *w1, float *w2, float *r)
    {
        const int i = threadIdx.x;
        f_out[i] = (w1[i]*w2[i])/(r[i]*r[i]);
    }
            """)


logging.basicConfig(level=logging.DEBUG)


class Body():
    def __init__(self, x=None, y=None, z=None, weight=None):
        assert all([isinstance(v, float) for v in [x, y, z, weight]])
        self.x = x
        self.y = y
        self.z = z
        self.weight = weight

    @classmethod
    def make_random(cls):
        return cls(x=random(), y=random(), z=random(), weight=random())

    @classmethod
    def make_random_n(cls, n=10):
        bodies = [cls.make_random() for i in range(n)]
        return bodies



def dev_info():
    dev = drv.Device(0)
    ctx = dev.make_context()
    dev_attrs = attrs = ctx.get_device().get_attributes()
    dev_data = pycuda.tools.DeviceData()
    print("CUDA device info")
    print(" - Max threads: {}".format(dev_data.max_threads))
    print(" - Thread blocks per mp: {}".format(dev_data.thread_blocks_per_mp))
    print(" - Shared memory: {}".format(dev_data.shared_memory))
    print("")
    ctx.pop()
dev_info()


def timer(times=1):
    def dec(f, *args, **kwargs):
        def g(self):
            start = time()
            for n in range(times):
                f(self, *args, **kwargs)
            time_taken = time() - start
            print("""# Function {} ran {} times
    - Total time: {}s
    - Average time: {}s""".format(f, times, time_taken, time_taken/times))
        return g
    return dec

def newtons_law_gpu(w1, w2, r):
    out = numpy.zeros_like(w1)
    f = mod.get_function("newtons_law")
    f(drv.Out(out), drv.In(w1), drv.In(w2), drv.In(r), block=(1024,1,1), grid=(1,1))
    # TODO: Why is it off by ~150x?
    print((w1[0] * w2[0]) / (r[0]**2)/142)
    print(out[0])
    return out

def newtons_law_cpu(w1, w2, r):
    return w1 * w2 / (r**2)
    #return numpy.array([w1[i] * w2[i] / (r[i]**2) for i in range(len(w1))])

def bodies_to_newton(bodies):
    # bodies is a 4-tuple (x, y, z, mass)
    body_pairs = list(itertools.combinations(bodies, r=2))
    w1 = numpy.zeros(len(body_pairs))
    w2 = numpy.zeros(len(body_pairs))
    r = numpy.zeros(len(body_pairs))
    for i, pair in enumerate(body_pairs):
        b1, b2 = pair
        w1[i] = b1.weight
        w2[i] = b2.weight
        r[i] = math.pow((b1.x - b2.x)**2 + (b1.y - b2.y)**2 + (b1.z-b2.z)**2, 1/3)
    return w1, w2, r

def newtons_law(w1, w2, r, use_gpu=True):
    if use_gpu:
        out = newtons_law_gpu(w1, w2, r)
    else:
        out = newtons_law_cpu(w1, w2, r)

    return out


class NewtonsTests(unittest.TestCase):
    # TODO: Fix CPU/GPU not giving the same answer.

    n = 4
    bodies = Body.make_random_n(n=n)

    def setUp(self):
        self.w1, self.w2, self.r = bodies_to_newton(self.bodies)

    @timer(times=1)
    def test_benchmark_bodies_to_newton(self):
        bodies_to_newton(Body.make_random_n(n=self.n))

    @timer(times=1)
    def test_benchmark_gpu(self):
        out = newtons_law(self.w1, self.w2, self.r)
        logging.debug(out[-10:])

    @timer(times=1)
    def test_benchmark_cpu(self):
        out = newtons_law(self.w1, self.w2, self.r, use_gpu=False)
        logging.debug(out[-10:])


if __name__ == "__main__":
    unittest.main()
