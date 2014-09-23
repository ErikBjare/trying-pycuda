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

    __global__ void multiply_them(float *dest, float *a, float *b)
    {
        const int i = threadIdx.x;
        dest[i] = a[i] * b[i];
    }
            """)


logging.basicConfig(level=logging.INFO)

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


def multiply_rands():
    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)
    dest = numpy.zeros_like(a)
    multiply_them(drv.Out(dest), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))

    print(dest)
    print(dest-a*b)

def newtons_law_gpu(w1, w2, r):
    out = numpy.zeros_like(w1)
    f = mod.get_function("newtons_law")
    f(drv.Out(out), drv.In(w1), drv.In(w2), drv.In(r), block=(128,1,1), grid=(4,4))
    return out

def newtons_law_cpu(w1, w2, r):
    return numpy.array([w1[i] * w2[i] / (r[i]**2) for i in range(len(w1))])

def bodies_to_newton(bodies):
    # bodies is a 4-tuple (x, y, z, mass)
    body_pairs = list(itertools.combinations(bodies, r=2))
    w1 = numpy.zeros(len(body_pairs))
    w2 = numpy.zeros(len(body_pairs))
    r = numpy.zeros(len(body_pairs))
    for i, pair in enumerate(body_pairs):
        b1, b2 = pair
        w1[i] = b1[3]
        w2[i] = b2[3]
        r[i] = math.pow((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2 + (b1[2]-b2[2])**2, 1/3)
    return w1, w2, r

def newtons_law(w1, w2, r, use_gpu=True):
    if use_gpu:
        out = newtons_law_gpu(w1, w2, r)
    else:
        out = newtons_law_cpu(w1, w2, r)

    return out


def random_bodies(n=10):
    bodies = []
    for i in range(n):
        bodies.append((random(), random(), random(), random()))
    return bodies




class NewtonsTests(unittest.TestCase):
    def setUp(self, n=500):
        self.n = n
        self.w1, self.w2, self.r = bodies_to_newton(random_bodies(n=n))

    @timer(times=10)
    def test_benchmark_bodies_to_newton(self):
        bodies_to_newton(random_bodies(n=self.n))

    @timer(times=10)
    def test_benchmark_gpu(self):
        out = newtons_law(self.w1, self.w2, self.r)
        logging.debug(out[:10])


    @timer(times=5)
    def test_benchmark_cpu(self):
        out = newtons_law(self.w1, self.w2, self.r, use_gpu=False)
        logging.debug(out[:10])


if __name__ == "__main__":
    unittest.main()
