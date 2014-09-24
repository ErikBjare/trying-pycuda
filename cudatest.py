import logging
from time import time
import math
from random import randint, random
import itertools
import unittest

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
    __global__ void newtons_law(float *f_out, float *w1, float *w2, float *r)
    {
        const int i = blockIdx.x*1024 + threadIdx.x;
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
    # TODO: It is still slower than CPU when it should outperform, investigate!
    # Perhaps NumPy already utilizes the GPU?

    out = numpy.zeros_like(w1)
    # TODO: GPUArray works, but is much slower than the CPU equivalent and my own kernel.
    # Figure out why!
    #w1 = pycuda.gpuarray.to_gpu(w1)
    #w2 = pycuda.gpuarray.to_gpu(w2)
    #r = pycuda.gpuarray.to_gpu(r)
    #out = w1*w2/(r**2)
    f = mod.get_function("newtons_law")

    block = (1024, 1, 1)
    grid = (math.floor(len(out)/1024+1), 1)
    threads = block[0] * block[1] * block[2] * grid[0] * grid[1]
    print("Threads: {}, Out: {}".format(threads, len(out)))

    f(drv.Out(out), drv.In(w1), drv.In(w2), drv.In(r), block=block, grid=grid)
    return out

def newtons_law_cpu(w1, w2, r):
    return w1 * w2 / (r**2)

def bodies_to_newton(bodies):
    body_pairs = list(itertools.combinations(bodies, r=2))
    w1 = numpy.zeros(len(body_pairs)).astype(numpy.float32)
    w2 = numpy.zeros(len(body_pairs)).astype(numpy.float32)
    r = numpy.zeros(len(body_pairs)).astype(numpy.float32)
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
    n = 100
    bodies = Body.make_random_n(n=n)

    def setUp(self):
        self.w1, self.w2, self.r = bodies_to_newton(self.bodies)

    @timer(times=1)
    def test_benchmark_bodies_to_newton(self):
        bodies_to_newton(Body.make_random_n(n=self.n))

    @timer(times=1)
    def test_benchmark_gpu(self):
        out = newtons_law(self.w1, self.w2, self.r)

    @timer(times=1)
    def test_benchmark_cpu(self):
        out = newtons_law(self.w1, self.w2, self.r, use_gpu=False)

    def test_equiv_implementations(self):
        out_gpu = newtons_law(self.w1, self.w2, self.r, use_gpu=True)
        out_cpu = newtons_law(self.w1, self.w2, self.r, use_gpu=False)
        self.assertTrue((out_gpu[-10:] == out_cpu[-10:]).all(), "GPU and CPU implementations gave different answers")



if __name__ == "__main__":
    unittest.main()
