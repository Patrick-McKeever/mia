import json
import numpy as np
from scipy.stats import poisson, nbinom
from abc import ABC, abstractmethod


class CountDistribution(ABC):
    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def variance(self):
        pass

    @abstractmethod
    def pmf(self):
        pass


class Poisson(CountDistribution):
    def __init__(self, lam):
        self.lam = lam

    def mean(self):
        return self.lam

    def variance(self):
        return self.lam

    def to_dict(self):
        return {"type": "Poisson", "lam": self.lam.item()}

    def pmf(self, k):
        return poisson.pmf(k, mu=self.lam)

    def pmf_vec(self, k):
        return poisson.pmf(k, mu=self.lam)

    @classmethod
    def from_dict(cls, d):
        return cls(lam=d["lam"])


class ZeroInflatedPoisson(CountDistribution):
    def __init__(self, pi, lam):
        self.pi = pi
        self.lam = lam

    def mean(self):
        return (1 - self.pi) * self.lam

    def variance(self):
        return (1 - self.pi) * (self.lam + self.pi * self.lam ** 2)

    def to_dict(self):
        return {"type": "ZeroInflatedPoisson", "pi": self.pi.item(), "lam": self.lam.item()}

    def pmf(self, k):
        poisson_part = poisson.pmf(k, mu=self.lam)
        if k == 0:
            return self.pi + (1 - self.pi) * poisson_part
        else:
            return (1 - self.pi) * poisson_part

    def pmf_vec(self, k):
        k = np.asarray(k)
        nb_part = nbinom.pmf(k, n=self.r, p=self.p)
        mask = (k == 0)
        return np.where(
            mask,
            self.pi + (1 - self.pi) * nb_part,
            (1 - self.pi) * nb_part
        )

    @classmethod
    def from_dict(cls, d):
        return cls(pi=d["pi"], lam=d["lam"])


class NegativeBinomial(CountDistribution):
    def __init__(self, r, p):
        self.r = r
        self.p = p

    def mean(self):
        return self.r * (1 - self.p) / self.p

    def variance(self):
        return self.r * (1 - self.p) / (self.p ** 2)

    def to_dict(self):
        return {"type": "NegativeBinomial", "r": self.r.item(), "p": self.p.item()}

    def pmf(self, k):
        return nbinom.pmf(k, n=self.r, p=self.p)

    def pmf_vec(self, k):
        return nbinom.pmf(k, n=self.r, p=self.p)

    @classmethod
    def from_dict(cls, d):
        return cls(r=d["r"], p=d["p"])


class ZeroInflatedNegativeBinomial(CountDistribution):
    def __init__(self, pi, r, p):
        self.pi = pi
        self.r = r
        self.p = p

    def mean(self):
        base_mean = self.r * (1 - self.p) / self.p
        return (1 - self.pi) * base_mean

    def variance(self):
        base_mean = self.r * (1 - self.p) / self.p
        base_var = self.r * (1 - self.p) / (self.p ** 2)
        return (1 - self.pi) * (base_var + self.pi * base_mean ** 2)

    def pmf(self, k):
        nb_part = nbinom.pmf(k, n=self.r, p=self.p)
        if k == 0:
            return self.pi + (1 - self.pi) * nb_part
        else:
            return (1 - self.pi) * nb_part

    def pmf_vec(self, k):
        k = np.asarray(k)
        nb_part = nbinom.pmf(k, n=self.r, p=self.p)
        mask = (k == 0)
        return np.where(
            mask,
            self.pi + (1 - self.pi) * nb_part,
            (1 - self.pi) * nb_part
        )

    def to_dict(self):
        return {"type": "ZeroInflatedNegativeBinomial", "pi": self.pi.item(), "r": self.r.item(), "p": self.p.item()}

    @classmethod
    def from_dict(cls, d):
        return cls(pi=d["pi"], r=d["r"], p=d["p"])


def save_distribution_to_json(dist, filename):
    with open(filename, "w") as f:
        json.dump(dist.to_dict(), f, indent=2)


def load_distribution_from_dict(d):
    dist_type = d.get("type")
    classes = {
        "Poisson": Poisson,
        "ZeroInflatedPoisson": ZeroInflatedPoisson,
        "NegativeBinomial": NegativeBinomial,
        "ZeroInflatedNegativeBinomial": ZeroInflatedNegativeBinomial
    }
    cls = classes.get(dist_type)
    if cls is None:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    return cls.from_dict(d)


def load_distribution_from_json(filename):
    with open(filename, "r") as f:
        d = json.load(f)
    return load_distribution_from_dict(d)

