import numpy as np
import jax.numpy as jaxnp
from jax import grad, jit
from functools import partial


class CNModel:
    """MLIAP-compatible entropy model using JAX for automatic differentiation.

    Computes the negative log-determinant of the normalized information matrix
    built from SNAP bispectrum descriptors. When used as a LAMMPS MLIAP model,
    provides energy and forces (beta) that drive atomic relaxation toward
    configurations that maximize information entropy.

    Supports descriptor masking via the mask parameter, which selects a subset
    of descriptors from the full descriptor space. When mask covers all
    descriptors, this is equivalent to using the full space (binary case).
    """

    def __init__(self, n_elements, n_descriptors_tot, energy_mode=True,
                 populations=None, mask=None, cross_=None, renorm_=None,
                 mean_=None, count_=0, epsilon_=1e-6):
        self.n_params = 1  # required by MLIAPPY
        self.n_elements = n_elements
        self.epsilon = epsilon_
        self.n_descriptors = n_descriptors_tot
        self.energy_mode = energy_mode
        self.active = True
        self.count = count_

        if mask is None:
            mask = list(range(n_descriptors_tot))
        self.mask = mask
        self.n_descriptors_keep = len(self.mask)

        if self.count == 0:
            self.active = False
            self.renorm = None
            self.mean = None
            self.cross = None
            return
        else:
            self.renorm = renorm_[mask, :][:, mask]
            self.mean = mean_[mask]
            self.cross = cross_[mask, :][:, mask]

        if self.renorm is None:
            self.renorm = np.identity(self.n_descriptors_keep)
        if self.mean is None:
            self.mean = np.zeros(self.n_descriptors_keep)

        self.populations = populations
        self.reg = self.epsilon * np.identity(self.n_descriptors_keep)
        self.cn_grad = grad(self.cn)
        self.K = 1

    @partial(jit, static_argnums=(0,))
    def cn(self, descriptors):
        d = descriptors - self.mean
        if self.energy_mode:
            d = jaxnp.mean(descriptors, axis=0)
            d = d.reshape((1, -1))
        if self.active:
            effective_count = self.count + d.shape[0]
            information = (self.cross + d.T @ d) / effective_count
            projected_information = jaxnp.divide(information, self.renorm) + self.reg
            (sign, logabsdet) = jaxnp.linalg.slogdet(projected_information)
            return -logabsdet
        else:
            return 0

    def __call__(self, elems, bispectrum, beta, energy):
        self.last_bispectrum = bispectrum.copy()
        b = bispectrum[:, self.mask]

        if self.active:
            energy[:] = 0
            energy[0] = self.K * self.cn(b)
            b = self.K * self.cn_grad(b)
            beta[:, :] = 0
            beta[:, self.mask] = b
            if not jaxnp.all(jaxnp.isfinite(b)):
                print("GRAD ERROR!")
        else:
            energy[:] = 0
            beta[:, :] = 0

        # Cleanup JAX cache to prevent unbounded memory growth
        if self.cn._cache_size() > 30:
            self.cn._clear_cache()
            import gc
            import sys
            for module_name, module in list(sys.modules.items()):
                if module_name.startswith("jax"):
                    if module_name not in ["jax.interpreters.partial_eval"]:
                        for obj_name in dir(module):
                            obj = getattr(module, obj_name)
                            if hasattr(obj, "cache_clear"):
                                try:
                                    obj.cache_clear()
                                except Exception:
                                    pass
            gc.collect()


class CNManager:
    """Tracks descriptor statistics for entropy evaluation.

    Accumulates the mean-subtracted sum and cross-product matrices across
    configurations. Used to evaluate the current information entropy and to
    tentatively evaluate candidate configurations before accepting them.
    """

    def __init__(self, n_descriptors, epsilon=0, mean=None, renorm=None,
                 energy_mode=True):
        self.epsilon = epsilon
        self.count = 0
        self.n_descriptors = n_descriptors
        self.sum = np.zeros((self.n_descriptors,))
        self.cross = np.zeros((self.n_descriptors, self.n_descriptors))
        self.reg = epsilon * np.identity(self.n_descriptors)
        self.data = []
        self.s = None
        self.energy_mode = energy_mode

        self.mean = mean if mean is not None else np.zeros((self.n_descriptors,))
        self.renorm = renorm if renorm is not None else np.ones((self.n_descriptors, self.n_descriptors))

    def print_status(self):
        cond, det = self.evaluate()
        print("STATUS  -- COUNT ", self.count, " COND: ", cond, "DET: ", det,
              flush=True)

    def update(self, dd, key=None):
        self.data.append(dd)
        dt = dd - self.mean
        if self.energy_mode:
            dt = np.mean(dt, axis=0)
            dt = dt.reshape((1, -1))

        self.sum += np.sum(dt, axis=0)
        self.cross += dt.T @ dt
        self.count += dt.shape[0]

        information = self.cross / self.count
        projected_information = np.divide(information, self.renorm)
        projected_information += self.reg

        try:
            u, s, vh = np.linalg.svd(projected_information)
            self.s = s
        except Exception:
            pass

    def evaluate(self, dd=None, key=None):
        effective_count = self.count
        if dd is not None:
            dt = dd - self.mean
            if self.energy_mode:
                dt = np.mean(dt, axis=0)
                dt = dt.reshape((1, -1))
            cross = self.cross.copy()
            cross += dt.T @ dt
            effective_count += dt.shape[0]
            information = cross / effective_count
        else:
            information = self.cross / effective_count

        projected_information = np.divide(information, self.renorm) + self.reg
        self.projected_information = projected_information

        (sign, logabsdet) = jaxnp.linalg.slogdet(projected_information)
        return jaxnp.linalg.cond(projected_information), -logabsdet
