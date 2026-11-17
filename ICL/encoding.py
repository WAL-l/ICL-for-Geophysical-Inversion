from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ICL.config import InversionConfig

_INV_CONFIG = InversionConfig()


@dataclass(frozen=True, slots=True)
class ModelCodec:
    rho_min: float
    rho_max: float

    # def _encode(self, rho):
    #     lo, hi = np.log10([self.rho_min, self.rho_max])
    #     # code = (np.log10(rho) - lo) / (hi - lo) * MODEL_CODE_MAX
    #     return np.rint(code).astype(int)
    def encode_log_norm(self, rho):
        lo, hi = np.log10([self.rho_min, self.rho_max])
        code = (np.log10(rho) - lo) / (hi - lo)
        return code

    def encode_log(self, rho):
        return np.log10(rho)

    def decode_log_norm(self, values):
        lo, hi = np.log10([self.rho_min, self.rho_max])
        return np.power(10.0, lo + values * (hi - lo))

    def decode_log(self, values):
        return np.power(10.0, values)


@dataclass(frozen=True, slots=True)
class ResponseCodec:
    log_min: float
    log_max: float
    min: float
    max: float

    @classmethod
    def fit(cls,values):
        logs = np.log10(values)
        lo, hi = np.min(logs), np.max(logs)
        d_lo, d_hi = np.min(values), np.max(values)
        # if not hi > lo:
        #     hi = lo + 1.0
        # padding = 0.05 * (hi - lo)
        return cls(lo, hi, d_lo, d_hi)

    def token(self, z):
        return f"{int(z):04d}"

    def encode_str(self, values):
        return values

    def encode_log(self, values):
        logs = np.log10(values)
        return np.abs(logs)

    def encode_row_norm(self, values):
        scaled = (values - self.min) / (self.max - self.min)
        codes = np.rint(np.clip(scaled, 0.0, 1.0) * _INV_CONFIG.response_code_max).astype(int)
        return [self.token(z) for z in codes]

    def encode_norm(self, values):
        scaled = (values - self.min) / (self.max - self.min)
        return scaled

    def encode_row_log_norm(self, values):
        logs = np.log10(values)
        scaled = (logs - self.log_min) / (self.log_max - self.log_min)
        codes = np.rint(np.clip(scaled, 0.0, 1.0) * _INV_CONFIG.response_code_max).astype(int)
        return [self.token(z) for z in codes]


    def encode_log_norm(self, values):
        logs = np.log10(values)
        scaled = (logs - self.log_min) / (self.log_max - self.log_min)
        return scaled

