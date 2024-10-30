import numpy as np
import constants as cst

def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def Bgrid(size, x_min, x_max):

    x_grid = np.linspace(x_min, x_max, size+1)

    # Create a 3D grid
    X, Y, Z = np.meshgrid(x_grid, x_grid, x_grid)

    # Calculate r, r_magnitude for each point in the grid
    r = np.stack((Y, X, Z), axis=-1)  # Shape is (size, size, size, 3)
    r_magnitude = np.linalg.norm(r, axis=-1)

    # Calculate the magnetic field B_grids
    first_term = -3.07e-5*cst.R_e**3  / (r_magnitude ** 5)[..., np.newaxis]

    x = r[:, :, :, 0:1]
    y = r[:, :, :, 1:2]
    z = r[:, :, :, 2:3]

    second_term = np.concatenate((3*x*z, 3*y*z, (2*z**2-x**2-y**2)), axis=3)

    B_grid = first_term * second_term

    return B_grid

class Particle:

    def __init__(self, mass, charge, kinetic_energy, dt_size, B_func, dx=None, x_min=None, x_max=None, intepo_method='ana'):
        self.mass = mass
        self.charge = charge
        self.kinetic_energy = kinetic_energy
        self.gamma = 1 + (self.kinetic_energy * abs(self.charge)) / (self.mass * cst.c**2)
        self.dt_size = dt_size
        self.time = 0

        self.position = np.zeros((3), dtype='float64')
        self.v = np.zeros((3), dtype='float64')
        self.u = np.zeros((3), dtype='float64')

        self.B_func = B_func

        if intepo_method != 'ana':
            if x_min is None or x_max is None or dx is None:
                raise ValueError("x_min, x_max, and dx must be provided for grid-based interpolation.")
            self.dx = dx * cst.R_e
            self.x_min = x_min * cst.R_e
            self.x_max = x_max * cst.R_e
            size = int((self.x_max - self.x_min) / (self.dx))
            self.B_grid = Bgrid(size, self.x_min, self.x_max)

    def initPos(self, x, y, z):
        self.position[:] = (x, y, z)
        self.B, self.Bmod = self.B_func(self.position)
        self.delta_t()

    def init_theta(self, theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        v = np.array([0.0, np.sin(theta_rad), np.cos(theta_rad)])

        gamma_inv_sq = 1 / self.gamma**2
        vmag = cst.c * np.sqrt(1 - gamma_inv_sq)
        v *= vmag / np.linalg.norm(v) 

        self.v[:] = v
        self.u[:] = self.gamma * v
        self.parms()
    
    def delta_t(self):
        self.dt = self.mass * self.gamma / (abs(self.charge) * self.Bmod * self.dt_size)

    def parms(self):
        p = self.mass * self.u
        p_dot_B = np.dot(p, self.B)
        self.p_para = p_dot_B / self.Bmod
        self.p_perp2 = np.dot(p, p) - (self.p_para**2)
        self.mu = self.p_perp2 / (2 * self.mass * self.Bmod)

class Interpolation:
    @staticmethod
    def B_ana(x):

        r1 = np.linalg.norm(x) 
        r2 = dot(x, x)**2
        r_inv_5 = 1 / (r1 * r2)

        B = -3.07e-5 * cst.R_e**3 * r_inv_5 * np.array([
            3 * x[0] * x[2], 
            3 * x[1] * x[2], 
            2 * x[2]**2 - x[0]**2 - x[1]**2
        ])

        Bmod = np.linalg.norm(B)  

        return B, Bmod

    def tri(self, particle, x):
        # interpolate the a value at position x
        i = (x[0] - particle.x_min)/particle.dx
        j = (x[1] - particle.x_min)/particle.dx
        k = (x[2] - particle.x_min)/particle.dx

        i0 = int(np.floor(i))
        i1 = i0 + 1

        j0 = int(np.floor(j))
        j1 = j0 + 1

        k0 = int(np.floor(k))
        k1 = k0 + 1
        
        xfac = i - i0
        yfac = j - j0
        zfac = k - k0

        b000 = particle.B_grid[i0,j0,k0]
        b100 = particle.B_grid[i1,j0,k0] 
        b001 = particle.B_grid[i0,j0,k1]
        b010 = particle.B_grid[i0,j1,k0]
        b011 = particle.B_grid[i0,j1,k1]
        b101 = particle.B_grid[i1,j0,k1]
        b110 = particle.B_grid[i1,j1,k0]
        b111 = particle.B_grid[i1,j1,k1]

        b00 = b000*(1-xfac)+b100*xfac
        b01 = b001*(1-xfac)+b101*xfac
        b10 = b010*(1-xfac)+b110*xfac
        b11 = b011*(1-xfac)+b111*xfac

        b0 = b00*(1-yfac)+b10*yfac
        b1 = b01*(1-yfac)+b11*yfac

        b = b0*(1-zfac)+b1*zfac
        bmod = np.linalg.norm(b)  
        return b, bmod

    def tsc(self, particle, x):
        # interpolate the a value at position x
        i = (x[0] - particle.x_min)/particle.dx
        j = (x[1] - particle.x_min)/particle.dx
        k = (x[2] - particle.x_min)/particle.dx

        i2 = np.floor(i + 0.5).astype(int) 
        i1 = i2 - 1
        i3 = i2 + 1

        j2 = np.floor(j + 0.5).astype(int) 
        j1 = j2 - 1
        j3 = j2 + 1

        k2 = np.floor(k + 0.5).astype(int) 
        k1 = k2 - 1
        k3 = k2 + 1

        xfrac = i - i2
        yfrac = j - j2
        zfrac = k - k2

        b_111 = particle.B_grid[i1,j1,k1]
        b_112 = particle.B_grid[i1,j1,k2]
        b_113 = particle.B_grid[i1,j1,k3]
        b_121 = particle.B_grid[i1,j2,k1]
        b_122 = particle.B_grid[i1,j2,k2]
        b_123 = particle.B_grid[i1,j2,k3]
        b_131 = particle.B_grid[i1,j3,k1]
        b_132 = particle.B_grid[i1,j3,k2]
        b_133 = particle.B_grid[i1,j3,k3]

        b_211 = particle.B_grid[i2,j1,k1]
        b_212 = particle.B_grid[i2,j1,k2]
        b_213 = particle.B_grid[i2,j1,k3]
        b_221 = particle.B_grid[i2,j2,k1]
        b_222 = particle.B_grid[i2,j2,k2]
        b_223 = particle.B_grid[i2,j2,k3]
        b_231 = particle.B_grid[i2,j3,k1]
        b_232 = particle.B_grid[i2,j3,k2]
        b_233 = particle.B_grid[i2,j3,k3]

        b_311 = particle.B_grid[i3,j1,k1]
        b_312 = particle.B_grid[i3,j1,k2]
        b_313 = particle.B_grid[i3,j1,k3]
        b_321 = particle.B_grid[i3,j2,k1]
        b_322 = particle.B_grid[i3,j2,k2]
        b_323 = particle.B_grid[i3,j2,k3]
        b_331 = particle.B_grid[i3,j3,k1]
        b_332 = particle.B_grid[i3,j3,k2]
        b_333 = particle.B_grid[i3,j3,k3]

        xw2 = 3/4 - (xfrac)**2
        xw1 = 1/2 * (1/2 - xfrac)**2
        xw3 = 1/2 * (1/2 + xfrac)**2

        b11 = b_211 * xw2 + b_111 * xw1 + b_311 * xw3
        b12 = b_212 * xw2 + b_112 * xw1 + b_312 * xw3
        b13 = b_213 * xw2 + b_113 * xw1 + b_313 * xw3
        b21 = b_221 * xw2 + b_121 * xw1 + b_321 * xw3
        b22 = b_222 * xw2 + b_122 * xw1 + b_322 * xw3
        b23 = b_223 * xw2 + b_123 * xw1 + b_323 * xw3
        b31 = b_231 * xw2 + b_131 * xw1 + b_331 * xw3
        b32 = b_232 * xw2 + b_132 * xw1 + b_332 * xw3
        b33 = b_233 * xw2 + b_133 * xw1 + b_333 * xw3

        yw2 = 3/4 - (yfrac)**2
        yw1 = 1/2 * (1/2 - yfrac)**2
        yw3 = 1/2 * (1/2 + yfrac)**2

        b1 = b11 * yw1 + b21 * yw2 + b31 * yw3
        b2 = b12 * yw1 + b22 * yw2 + b32 * yw3
        b3 = b13 * yw1 + b23 * yw2 + b33 * yw3

        zw2 = 3/4 - (zfrac)**2
        zw1 = 1/2 * (1/2 - zfrac)**2
        zw3 = 1/2 * (1/2 + zfrac)**2

        b = b1 * zw1 + b2 * zw2 + b3 * zw3
        bmod = np.linalg.norm(b)  

        return b, bmod

    def bspline(self, particle, x):

        i = (x[0] - particle.x_min)/particle.dx
        j = (x[1] - particle.x_min)/particle.dx
        k = (x[2] - particle.x_min)/particle.dx

        i0 = np.floor(i+0.5).astype(int)
        j0 = np.floor(j+0.5).astype(int)
        k0 = np.floor(k+0.5).astype(int)

        xfrac = i0 - i
        yfrac = j0 - j
        zfrac = k0 - k

        i_m2, i_m1, i_p1, i_p2 = i0 - 2, i0 - 1, i0 + 1, i0 + 2
        j_m2, j_m1, j_p1, j_p2 = j0 - 2, j0 - 1, j0 + 1, j0 + 2
        k_m2, k_m1, k_p1, k_p2 = k0 - 2, k0 - 1, k0 + 1, k0 + 2

        third = 1/3
        fac1 = 0.125*third
        fac2 = 0.5*third
        fac3 = 7.1875*third

        # Weight factors
        cx2 = (xfrac)**2
        gx_m2 = fac1* (0.5 + xfrac)**4
        gx_m1 = fac2*(1.1875 + 2.75*xfrac + cx2*(1.5 - xfrac - cx2))
        gx_0 = 0.25 * (fac3 + cx2 * (cx2 - 2.5))
        gx_p1 = fac2*(1.1875 - 2.75*xfrac + cx2*(1.5 + xfrac - cx2))
        gx_p2 = fac1*(0.5 - xfrac)**4

        cy2 = yfrac**2
        gy_m2 = fac1*(0.5 + yfrac)**4
        gy_m1 = fac2*(1.1875 + 2.75*yfrac + cy2*(1.5 - yfrac - cy2))
        gy_0 = 0.25 * (fac3 + cy2 * (cy2 - 2.5))
        gy_p1 = fac2*(1.1875 - 2.75*yfrac + cy2*(1.5 + yfrac - cy2))
        gy_p2 = fac1*(0.5 - yfrac)**4

        cz2 = zfrac**2
        gz_m2 = fac1*(0.5 + zfrac)**4
        gz_m1 = fac2*(1.1875 + 2.75*zfrac + cz2*(1.5 - zfrac - cz2))
        gz_0 = 0.25 * (fac3 + cz2 * (cz2 - 2.5))
        gz_p1 = fac2*(1.1875 - 2.75*zfrac + cz2*(1.5 + zfrac - cz2))
        gz_p2 = fac1*(0.5 - zfrac)**4

        b_111 = particle.B_grid[i_m2, j_m2, k_m2]
        b_112 = particle.B_grid[i_m2, j_m2, k_m1]
        b_113 = particle.B_grid[i_m2, j_m2, k0]
        b_114 = particle.B_grid[i_m2, j_m2, k_p1]
        b_115 = particle.B_grid[i_m2, j_m2, k_p2]

        b_121 = particle.B_grid[i_m2, j_m1, k_m2]
        b_122 = particle.B_grid[i_m2, j_m1, k_m1]
        b_123 = particle.B_grid[i_m2, j_m1, k0]
        b_124 = particle.B_grid[i_m2, j_m1, k_p1]
        b_125 = particle.B_grid[i_m2, j_m1, k_p2]

        b_131 = particle.B_grid[i_m2, j0, k_m2]
        b_132 = particle.B_grid[i_m2, j0, k_m1]
        b_133 = particle.B_grid[i_m2, j0, k0]
        b_134 = particle.B_grid[i_m2, j0, k_p1]
        b_135 = particle.B_grid[i_m2, j0, k_p2]

        b_141 = particle.B_grid[i_m2, j_p1, k_m2]
        b_142 = particle.B_grid[i_m2, j_p1, k_m1]
        b_143 = particle.B_grid[i_m2, j_p1, k0]
        b_144 = particle.B_grid[i_m2, j_p1, k_p1]
        b_145 = particle.B_grid[i_m2, j_p1, k_p2]

        b_151 = particle.B_grid[i_m2, j_p2, k_m2]
        b_152 = particle.B_grid[i_m2, j_p2, k_m1]
        b_153 = particle.B_grid[i_m2, j_p2, k0]
        b_154 = particle.B_grid[i_m2, j_p2, k_p1]
        b_155 = particle.B_grid[i_m2, j_p2, k_p2]

        b_211 = particle.B_grid[i_m1, j_m2, k_m2]
        b_212 = particle.B_grid[i_m1, j_m2, k_m1]
        b_213 = particle.B_grid[i_m1, j_m2, k0]
        b_214 = particle.B_grid[i_m1, j_m2, k_p1]
        b_215 = particle.B_grid[i_m1, j_m2, k_p2]

        b_221 = particle.B_grid[i_m1, j_m1, k_m2]
        b_222 = particle.B_grid[i_m1, j_m1, k_m1]
        b_223 = particle.B_grid[i_m1, j_m1, k0]
        b_224 = particle.B_grid[i_m1, j_m1, k_p1]
        b_225 = particle.B_grid[i_m1, j_m1, k_p2]

        b_231 = particle.B_grid[i_m1, j0, k_m2]
        b_232 = particle.B_grid[i_m1, j0, k_m1]
        b_233 = particle.B_grid[i_m1, j0, k0]
        b_234 = particle.B_grid[i_m1, j0, k_p1]
        b_235 = particle.B_grid[i_m1, j0, k_p2]

        b_241 = particle.B_grid[i_m1, j_p1, k_m2]
        b_242 = particle.B_grid[i_m1, j_p1, k_m1]
        b_243 = particle.B_grid[i_m1, j_p1, k0]
        b_244 = particle.B_grid[i_m1, j_p1, k_p1]
        b_245 = particle.B_grid[i_m1, j_p1, k_p2]

        b_251 = particle.B_grid[i_m1, j_p2, k_m2]
        b_252 = particle.B_grid[i_m1, j_p2, k_m1]
        b_253 = particle.B_grid[i_m1, j_p2, k0]
        b_254 = particle.B_grid[i_m1, j_p2, k_p1]
        b_255 = particle.B_grid[i_m1, j_p2, k_p2]

        b_311 = particle.B_grid[i0, j_m2, k_m2]
        b_312 = particle.B_grid[i0, j_m2, k_m1]
        b_313 = particle.B_grid[i0, j_m2, k0]
        b_314 = particle.B_grid[i0, j_m2, k_p1]
        b_315 = particle.B_grid[i0, j_m2, k_p2]

        b_321 = particle.B_grid[i0, j_m1, k_m2]
        b_322 = particle.B_grid[i0, j_m1, k_m1]
        b_323 = particle.B_grid[i0, j_m1, k0]
        b_324 = particle.B_grid[i0, j_m1, k_p1]
        b_325 = particle.B_grid[i0, j_m1, k_p2]

        b_331 = particle.B_grid[i0, j0, k_m2]
        b_332 = particle.B_grid[i0, j0, k_m1]
        b_333 = particle.B_grid[i0, j0, k0]
        b_334 = particle.B_grid[i0, j0, k_p1]
        b_335 = particle.B_grid[i0, j0, k_p2]

        b_341 = particle.B_grid[i0, j_p1, k_m2]
        b_342 = particle.B_grid[i0, j_p1, k_m1]
        b_343 = particle.B_grid[i0, j_p1, k0]
        b_344 = particle.B_grid[i0, j_p1, k_p1]
        b_345 = particle.B_grid[i0, j_p1, k_p2]

        b_351 = particle.B_grid[i0, j_p2, k_m2]
        b_352 = particle.B_grid[i0, j_p2, k_m1]
        b_353 = particle.B_grid[i0, j_p2, k0]
        b_354 = particle.B_grid[i0, j_p2, k_p1]
        b_355 = particle.B_grid[i0, j_p2, k_p2]

        b_411 = particle.B_grid[i_p1, j_m2, k_m2]
        b_412 = particle.B_grid[i_p1, j_m2, k_m1]
        b_413 = particle.B_grid[i_p1, j_m2, k0]
        b_414 = particle.B_grid[i_p1, j_m2, k_p1]
        b_415 = particle.B_grid[i_p1, j_m2, k_p2]

        b_421 = particle.B_grid[i_p1, j_m1, k_m2]
        b_422 = particle.B_grid[i_p1, j_m1, k_m1]
        b_423 = particle.B_grid[i_p1, j_m1, k0]
        b_424 = particle.B_grid[i_p1, j_m1, k_p1]
        b_425 = particle.B_grid[i_p1, j_m1, k_p2]

        b_431 = particle.B_grid[i_p1, j0, k_m2]
        b_432 = particle.B_grid[i_p1, j0, k_m1]
        b_433 = particle.B_grid[i_p1, j0, k0]
        b_434 = particle.B_grid[i_p1, j0, k_p1]
        b_435 = particle.B_grid[i_p1, j0, k_p2]

        b_441 = particle.B_grid[i_p1, j_p1, k_m2]
        b_442 = particle.B_grid[i_p1, j_p1, k_m1]
        b_443 = particle.B_grid[i_p1, j_p1, k0]
        b_444 = particle.B_grid[i_p1, j_p1, k_p1]
        b_445 = particle.B_grid[i_p1, j_p1, k_p2]

        b_451 = particle.B_grid[i_p1, j_p2, k_m2]
        b_452 = particle.B_grid[i_p1, j_p2, k_m1]
        b_453 = particle.B_grid[i_p1, j_p2, k0]
        b_454 = particle.B_grid[i_p1, j_p2, k_p1]
        b_455 = particle.B_grid[i_p1, j_p2, k_p2]

        b_511 = particle.B_grid[i_p2, j_m2, k_m2]
        b_512 = particle.B_grid[i_p2, j_m2, k_m1]
        b_513 = particle.B_grid[i_p2, j_m2, k0]
        b_514 = particle.B_grid[i_p2, j_m2, k_p1]
        b_515 = particle.B_grid[i_p2, j_m2, k_p2]

        b_521 = particle.B_grid[i_p2, j_m1, k_m2]
        b_522 = particle.B_grid[i_p2, j_m1, k_m1]
        b_523 = particle.B_grid[i_p2, j_m1, k0]
        b_524 = particle.B_grid[i_p2, j_m1, k_p1]
        b_525 = particle.B_grid[i_p2, j_m1, k_p2]

        b_531 = particle.B_grid[i_p2, j0, k_m2]
        b_532 = particle.B_grid[i_p2, j0, k_m1]
        b_533 = particle.B_grid[i_p2, j0, k0]
        b_534 = particle.B_grid[i_p2, j0, k_p1]
        b_535 = particle.B_grid[i_p2, j0, k_p2]

        b_541 = particle.B_grid[i_p2, j_p1, k_m2]
        b_542 = particle.B_grid[i_p2, j_p1, k_m1]
        b_543 = particle.B_grid[i_p2, j_p1, k0]
        b_544 = particle.B_grid[i_p2, j_p1, k_p1]
        b_545 = particle.B_grid[i_p2, j_p1, k_p2]

        b_551 = particle.B_grid[i_p2, j_p2, k_m2]
        b_552 = particle.B_grid[i_p2, j_p2, k_m1]
        b_553 = particle.B_grid[i_p2, j_p2, k0]
        b_554 = particle.B_grid[i_p2, j_p2, k_p1]
        b_555 = particle.B_grid[i_p2, j_p2, k_p2]
        
        b11 = b_111 * gx_m2 + b_211 * gx_m1 + b_311 * gx_0 + b_411 * gx_p1 + b_511 * gx_p2
        b21 = b_121 * gx_m2 + b_221 * gx_m1 + b_321 * gx_0 + b_421 * gx_p1 + b_521 * gx_p2
        b31 = b_131 * gx_m2 + b_231 * gx_m1 + b_331 * gx_0 + b_431 * gx_p1 + b_531 * gx_p2
        b41 = b_141 * gx_m2 + b_241 * gx_m1 + b_341 * gx_0 + b_441 * gx_p1 + b_541 * gx_p2
        b51 = b_151 * gx_m2 + b_251 * gx_m1 + b_351 * gx_0 + b_451 * gx_p1 + b_551 * gx_p2

        b12 = b_112 * gx_m2 + b_212 * gx_m1 + b_312 * gx_0 + b_412 * gx_p1 + b_512 * gx_p2
        b22 = b_122 * gx_m2 + b_222 * gx_m1 + b_322 * gx_0 + b_422 * gx_p1 + b_522 * gx_p2
        b32 = b_132 * gx_m2 + b_232 * gx_m1 + b_332 * gx_0 + b_432 * gx_p1 + b_532 * gx_p2
        b42 = b_142 * gx_m2 + b_242 * gx_m1 + b_342 * gx_0 + b_442 * gx_p1 + b_542 * gx_p2
        b52 = b_152 * gx_m2 + b_252 * gx_m1 + b_352 * gx_0 + b_452 * gx_p1 + b_552 * gx_p2

        b13 = b_113 * gx_m2 + b_213 * gx_m1 + b_313 * gx_0 + b_413 * gx_p1 + b_513 * gx_p2
        b23 = b_123 * gx_m2 + b_223 * gx_m1 + b_323 * gx_0 + b_423 * gx_p1 + b_523 * gx_p2
        b33 = b_133 * gx_m2 + b_233 * gx_m1 + b_333 * gx_0 + b_433 * gx_p1 + b_533 * gx_p2
        b43 = b_143 * gx_m2 + b_243 * gx_m1 + b_343 * gx_0 + b_443 * gx_p1 + b_543 * gx_p2
        b53 = b_153 * gx_m2 + b_253 * gx_m1 + b_353 * gx_0 + b_453 * gx_p1 + b_553 * gx_p2

        b14 = b_114 * gx_m2 + b_214 * gx_m1 + b_314 * gx_0 + b_414 * gx_p1 + b_514 * gx_p2
        b24 = b_124 * gx_m2 + b_224 * gx_m1 + b_324 * gx_0 + b_424 * gx_p1 + b_524 * gx_p2
        b34 = b_134 * gx_m2 + b_234 * gx_m1 + b_334 * gx_0 + b_434 * gx_p1 + b_534 * gx_p2
        b44 = b_144 * gx_m2 + b_244 * gx_m1 + b_344 * gx_0 + b_444 * gx_p1 + b_544 * gx_p2
        b54 = b_154 * gx_m2 + b_254 * gx_m1 + b_354 * gx_0 + b_454 * gx_p1 + b_554 * gx_p2

        b15 = b_115 * gx_m2 + b_215 * gx_m1 + b_315 * gx_0 + b_415 * gx_p1 + b_515 * gx_p2
        b25 = b_125 * gx_m2 + b_225 * gx_m1 + b_325 * gx_0 + b_425 * gx_p1 + b_525 * gx_p2
        b35 = b_135 * gx_m2 + b_235 * gx_m1 + b_335 * gx_0 + b_435 * gx_p1 + b_535 * gx_p2
        b45 = b_145 * gx_m2 + b_245 * gx_m1 + b_345 * gx_0 + b_445 * gx_p1 + b_545 * gx_p2
        b55 = b_155 * gx_m2 + b_255 * gx_m1 + b_355 * gx_0 + b_455 * gx_p1 + b_555 * gx_p2

        b1 = b11 * gy_m2 + b21 * gy_m1 + b31 * gy_0 + b41 * gy_p1 + b51 * gy_p2
        b2 = b12 * gy_m2 + b22 * gy_m1 + b32 * gy_0 + b42 * gy_p1 + b52 * gy_p2
        b3 = b13 * gy_m2 + b23 * gy_m1 + b33 * gy_0 + b43 * gy_p1 + b53 * gy_p2
        b4 = b14 * gy_m2 + b24 * gy_m1 + b34 * gy_0 + b44 * gy_p1 + b54 * gy_p2
        b5 = b15 * gy_m2 + b25 * gy_m1 + b35 * gy_0 + b45 * gy_p1 + b55 * gy_p2

        b = b1 * gz_m2 + b2 * gz_m1 + b3 * gz_0 + b4 * gz_p1 + b5 * gz_p2
        bmod = np.linalg.norm(b)  

        return b, bmod


class Integrator:
    
    def Boris(self, particle):

        # First half step for position
        particle.position += particle.u * particle.dt * 0.5 / particle.gamma

        # Calculate B at the intermediate position
        particle.B, particle.Bmod = particle.B_func(particle.position)
        
        # Calculate t and s for velocity update
        t = particle.charge * particle.dt * particle.B / (2 * particle.mass * particle.gamma)
        s = 2 * t / (1 + dot(t, t))

        # Update velocity
        u_prime = particle.u + cross(particle.u, t)
        particle.u += cross(u_prime, s) 

        # Update gamma based on the new velocity
        particle.gamma = np.sqrt(1 + dot(particle.u, particle.u) / (cst.c**2))

        particle.position += + particle.u * particle.dt * 0.5 / particle.gamma

        particle.B, particle.Bmod = particle.B_func(particle.position)

        particle.delta_t()
        particle.parms() 
        particle.time  += particle.dt

    def HC(self, particle):

        # First half step for position
        particle.position += particle.u * particle.dt * 0.5 / particle.gamma

        # Calculate B at the intermediate position
        particle.B, particle.Bmod = particle.B_func(particle.position)

        tau = particle.charge * particle.dt * particle.B / (2 * particle.mass)
        u_star = dot(particle.u, tau) / cst.c
        sigma = particle.gamma **2 - dot(tau, tau)

        gamma_plus1 = np.sqrt(sigma**2 + 4*(dot(tau, tau) + u_star**2))
        gamma_plus = np.sqrt((sigma + gamma_plus1)/2)

        t = tau / gamma_plus
        s = 1 / (1 + dot(t,t))

        # Update velocity
        u_plus1 = particle.u + dot(particle.u,t) * t + cross(particle.u,t)
        u_plus = s * u_plus1 
        particle.u = u_plus + cross(u_plus,t)
        
        # Update gamma based on the new velocity
        particle.gamma = np.sqrt(1 + dot(particle.u, particle.u) / (cst.c**2))

        particle.position += + particle.u * particle.dt * 0.5 / particle.gamma

        particle.B, particle.Bmod = particle.B_func(particle.position)

        particle.delta_t()
        particle.parms() 
        particle.time  += particle.dt

    def RK4(self, particle):

        # Calculate the Runge-Kutta steps
        k1 = particle.charge * cross(particle.v, particle.B) / (particle.gamma * particle.mass)

        k2_v = particle.v + k1 *particle.dt / 2
        k2_x = particle.position + k2_v *particle.dt / 2
        particle.B, particle.Bmod = particle.B_func(k2_x)

        k2 = particle.charge * cross(k2_v, particle.B) / (particle.gamma * particle.mass)

        k3_v = particle.v + k2 *particle.dt / 2
        k3_x = particle.position + k3_v *particle.dt / 2
        particle.B, particle.Bmod = particle.B_func(k3_x)

        k3 = particle.charge * cross(k3_v, particle.B) / (particle.gamma * particle.mass)

        k4_v = particle.v + k3 *particle.dt
        k4_x = particle.position + k4_v * particle.dt
        particle.B, particle.Bmod = particle.B_func(k4_x)

        k4 = particle.charge * cross(k4_v, particle.B) / (particle.gamma * particle.mass)

        # Update velocity and position using RK4 method
        particle.position += particle.dt / 6 * (particle.v + 2 * k2_v + 2 * k3_v + k4_v)
        particle.v += particle.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        particle.B, particle.Bmod = particle.B_func(particle.position)

        particle.gamma = 1 / (np.sqrt(1 - dot(particle.v, particle.v) / cst.c**2))
        particle.u = particle.v * particle.gamma

        particle.delta_t()
        particle.parms() 
        particle.time  += particle.dt

    def RK8_cv(self, particle):

        # Calculate the Runge-Kutta steps
        k1 = particle.charge * cross(particle.v, particle.B) / (particle.gamma * particle.mass)
        
        k2_v = particle.v + cst.a21 * k1 * particle.dt
        k2_x = particle.position + cst.a21 * k2_v * particle.dt
        particle.B, particle.Bmod = particle.B_func(k2_x)

        k2 = particle.charge * cross(k2_v, particle.B) / (particle.gamma * particle.mass)

        k3_v = particle.v + (cst.a31*k1 + cst.a32*k2) * particle.dt
        k3_x = particle.position + + (cst.a31*k2_v + cst.a32*k3_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k3_x)

        k3 = particle.charge * cross(k3_v, particle.B) / (particle.gamma * particle.mass)

        k4_v = particle.v + (cst.a41*k1 + cst.a42*k2 + cst.a43*k3) * particle.dt
        k4_x = particle.position + (cst.a41*k2_v + cst.a42*k3_v + cst.a43*k4_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k4_x)

        k4 = particle.charge * cross(k4_v, particle.B) / (particle.gamma * particle.mass)

        # Update velocity and position using RK4 method
        k5_v = particle.v + (cst.a51*k1 + cst.a53*k3 + cst.a54*k4) * particle.dt
        k5_x = particle.position  +  (cst.a51*k2_v + cst.a53*k4_v + cst.a54*k5_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k5_x)

        k5 = particle.charge * cross(k5_v, particle.B) / (particle.gamma * particle.mass)

        k6_v = particle.v  + (cst.a61*k1 + cst.a63*k3 + cst.a64*k4 + cst.a65*k5) * particle.dt
        k6_x = particle.position  + (cst.a61*k2_v + cst.a63*k4_v + cst.a64*k5_v + cst.a65*k6_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k6_x)

        k6 = particle.charge * cross(k6_v, particle.B) / (particle.gamma * particle.mass)

        k7_v = particle.v  + (cst.a71*k1 + cst.a73*k3 + cst.a74*k4 + cst.a75*k5 + cst.a76*k6) * particle.dt
        k7_x = particle.position  + (cst.a71*k2_v + cst.a73*k4_v + cst.a74*k5_v + cst.a75*k6_v + cst.a76*k7_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k7_x)

        k7 = particle.charge * cross(k7_v, particle.B) / (particle.gamma * particle.mass)

        k8_v = particle.v  + (cst.a81*k1 + cst.a85*k5 + cst.a86*k6 + cst.a87*k7) * particle.dt
        k8_x = particle.position  + (cst.a81*k2_v + cst.a85*k6_v + cst.a86*k7_v + cst.a87*k8_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k8_x)

        k8 = particle.charge * cross(k8_v, particle.B) / (particle.gamma * particle.mass)

        k9_v = particle.v  + (cst.a91*k1 + cst.a95*k5 + cst.a96*k6 + cst.a97*k7 + cst.a98*k8) * particle.dt
        k9_x = particle.position  + (cst.a91*k2_v + cst.a95*k6_v + cst.a96*k7_v + cst.a97*k8_v + cst.a98*k9_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k9_x)

        k9 = particle.charge * cross(k9_v, particle.B) / (particle.gamma * particle.mass)

        k10_v = particle.v  + (cst.a101*k1 + cst.a105*k5 + cst.a106*k6 + cst.a107*k7 + cst.a108*k8 + cst.a109*k9) * particle.dt
        k10_x = particle.position  + (cst.a101*k2_v + cst.a105*k6_v + cst.a106*k7_v + cst.a107*k8_v + cst.a108*k9_v + cst.a109*k10_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k10_x)

        k10 = particle.charge * cross(k10_v, particle.B) / (particle.gamma * particle.mass)

        k11_v = particle.v  + ( cst.a115*k5 + cst.a116*k6 + cst.a117*k7 + cst.a118*k8 + cst.a119*k9 + cst.a1110*k10) * particle.dt
        k11_x = particle.position  + ( cst.a115*k6_v + cst.a116*k7_v + cst.a117*k8_v + cst.a118*k9_v + cst.a119*k10_v + cst.a1110*k11_v) * particle.dt
        particle.B, particle.Bmod = particle.B_func(k11_x)

        k11 = particle.charge * cross(k11_v, particle.B) / (particle.gamma * particle.mass)

        # Update velocity and position using RK4 method
        particle.position += particle.dt *(cst.b1*particle.v + cst.b8*k8_v + cst.b9*k9_v + cst.b10*k10_v + cst.b11*k11_v)
        particle.v += particle.dt *(cst.b1*k1 + cst.b8*k8 + cst.b9*k9 + cst.b10*k10 + cst.b11*k11)

        particle.B, particle.Bmod = particle.B_func(particle.position)

        particle.gamma = 1 / (np.sqrt(1 - dot(particle.v, particle.v) / cst.c**2))
        particle.u = particle.v * particle.gamma

        particle.delta_t()
        particle.parms() 
        particle.time  += particle.dt

    def grad_B(self, particle, x):

        ip = np.array([x[0]+1.0, x[1], x[2]])
        im = np.array([x[0]-1.0, x[1], x[2]])

        jp = np.array([x[0], x[1]+1.0, x[2]])
        jm = np.array([x[0], x[1]-1.0, x[2]])

        kp = np.array([x[0], x[1], x[2]+1.0])
        km = np.array([x[0], x[1], x[2]-1.0])

        B_ip, Bmod_ip = particle.B_func(ip)
        bhat_ip = B_ip / Bmod_ip

        B_im, Bmod_im = particle.B_func(im)
        bhat_im = B_im / Bmod_im

        B_jp, Bmod_jp = particle.B_func(jp)
        bhat_jp = B_jp / Bmod_jp

        B_jm, Bmod_jm = particle.B_func(jm)
        bhat_jm = B_jm / Bmod_jm

        B_kp, Bmod_kp = particle.B_func(kp)
        bhat_kp = B_kp / Bmod_kp

        B_km, Bmod_km = particle.B_func(km)
        bhat_km = B_km / Bmod_km

        B, Bmod = particle.B_func(x)
        bhat = B / Bmod 

        gradBx = (Bmod_ip-Bmod_im)/2
        gradBy = (Bmod_jp-Bmod_jm)/2
        gradBz = (Bmod_kp-Bmod_km)/2

        gradB = np.array([gradBx, gradBy, gradBz])

        # dbhxdx = (bhat_ip[0] - bhat_im[0])/2
        dbhydx = (bhat_ip[1] - bhat_im[1])/2
        dbhzdx = (bhat_ip[2] - bhat_im[2])/2

        dbhxdy = (bhat_jp[0] - bhat_jm[0])/2
        # dbhydy = (bhat_jp[1] - bhat_jm[1])/2
        dbhzdy = (bhat_jp[2] - bhat_jm[2])/2

        dbhxdz = (bhat_kp[0] - bhat_km[0])/2
        dbhydz = (bhat_kp[1] - bhat_km[1])/2
        # dbhzdz = (bhat_kp[2] - bhat_km[2])/2

        curl_bhx = dbhzdy - dbhydz
        curl_bhy = dbhxdz - dbhzdx
        curl_bhz = dbhydx - dbhxdy

        curl_bhat = np.array([curl_bhx, curl_bhy, curl_bhz])

        return  B, bhat, gradB, curl_bhat

    def gca(self, particle, p_para, x):

        B, bhat, gradB, curl_bhat = self.grad_B(particle, x)

        B_star = B + (p_para/particle.charge)*curl_bhat   # vector

        B_star_par = dot(bhat, B_star)  # scalar

        E_star = -particle.mu*gradB/(particle.charge*particle.gamma)
        
        # relativistic guiding-centre velocity
        a = E_star * particle.charge
        b = B_star/B_star_par
        dt_p_para = dot(a,b)

        # relativistic guiding-centre parallel force
        dt_x_pos = p_para*B_star/(particle.mass*particle.gamma*B_star_par) + cross(E_star, bhat/B_star_par)

        return dt_p_para, dt_x_pos
    
    def gca_rk4(self, particle):

        k1p, k1x = self.gca(particle, particle.p_para, particle.position)

        x = particle.position + k1x*particle.dt/2
        p = particle.p_para + k1p*particle.dt/2

        k2p, k2x = self.gca(particle, p, x)

        x = particle.position + k2x*particle.dt/2
        p = particle.p_para + k2p*particle.dt/2

        k3p, k3x = self.gca(particle, p, x)

        x = particle.position + k3x*particle.dt
        p = particle.p_para + k3p*particle.dt

        k4p, k4x = self.gca(particle, p, x)

        particle.p_para += particle.dt/6*(k1p+2*k2p+2*k3p+k4p)
        particle.position += particle.dt/6*(k1x+2*k2x+2*k3x+k4x)

        particle.B, particle.Bmod = particle.B_func(particle.position)

        p_perp2 = 2*particle.mass*particle.Bmod*particle.mu
        u2 = (particle.p_para**2+p_perp2)/(particle.mass**2)
        particle.gamma = np.sqrt(1 + u2/cst.c**2)

        particle.delta_t()
        particle.time  += particle.dt


def push_particle(t_val, props, pusher, intepo_method, dx=None, x_min=None, x_max=None):
    # Dictionary mapping integrator methods to functions
    integrator_methods = {
        "boris": Integrator().Boris,
        "hc": Integrator().HC,
        "rk4": Integrator().RK4,
        "rk8": Integrator().RK8_cv,
        "gca": Integrator().gca_rk4,
    }

    integrator_method = integrator_methods[pusher]
    dt_size = 1 if pusher == 'gca' else 10

    intepo_methods = {
        "ana": lambda x: Interpolation().B_ana(x),
        "tri": lambda x: Interpolation().tri(part, x),
        "tsc": lambda x: Interpolation().tsc(part, x),
        "bsp": lambda x: Interpolation().bspline(part, x),
    }

    B_func = intepo_methods[intepo_method]

    part = Particle(props["mass"], props["charge"], props["kinetic_energy"], dt_size, B_func, dx, x_min, x_max, intepo_method)
    part.initPos(*props["position"])
    part.init_theta(props["theta_deg"])

    num_steps = int(t_val / part.dt)

    time = np.zeros(num_steps + 1, dtype='float64')
    traj = np.zeros((num_steps + 1, 3), dtype='float64')
    gamma = np.zeros(num_steps + 1, dtype='float64')
    mu = np.zeros(num_steps + 1, dtype='float64')
    p_para = np.zeros(num_steps + 1, dtype='float64')

    # Set initial values
    time[0] = part.time
    traj[0] = part.position
    gamma[0] = part.gamma
    mu[0] = part.mu
    p_para[0] = part.p_para

    # Run the simulation while time < t_val
    step = 1
    while part.time < t_val:
        integrator_method(part)
        if step <= num_steps:
            time[step] = part.time
            traj[step] = part.position
            gamma[step] = part.gamma
            mu[step] = part.mu
            p_para[step] = part.p_para
        step += 1
    return time, traj, gamma, mu, p_para
