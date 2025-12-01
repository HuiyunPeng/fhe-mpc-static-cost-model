import torch 
import numpy as np

from .tensors import CipherTensor

class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme 
        self.backend = scheme.backend
        self.new_polynomial_evaluator()
    
    def _log_primitive(self, primitive, **params):
        tracer = getattr(self.scheme, "trace_logger", None)
        if tracer:
            tracer.log_primitive(primitive, params)

    def new_polynomial_evaluator(self):
        self.backend.NewPolynomialEvaluator()

    def generate_monomial(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateMonomial(coeffs[::-1])
    
    def generate_chebyshev(self, coeffs):
        if isinstance(coeffs, (torch.Tensor, np.ndarray)):
            coeffs = coeffs.tolist()
        return self.backend.GenerateChebyshev(coeffs)

    def evaluate_polynomial(self, ciphertensor, poly, out_scale=None):
        out_scale = out_scale or self.scheme.params.get_default_scale()
        poly_depth = self.get_depth(poly)
        log_params = {"out_scale": out_scale}
        if poly_depth is not None:
            log_params["poly_depth"] = poly_depth
        self._log_primitive("PolyEval", **log_params)

        cts_out = []  
        for ctxt in ciphertensor.ids:
            ct_out = self.backend.EvaluatePolynomial(ctxt, poly, out_scale)
            cts_out.append(ct_out)

        return CipherTensor(
            self.scheme, cts_out, ciphertensor.shape, ciphertensor.on_shape)
    
    def generate_minimax_sign_coeffs(self, degrees, prec=128, logalpha=12, 
                                     logerr=12, debug=False):
        if isinstance(degrees, int):
            degrees = [degrees]
        else:
            degrees = list(degrees)

        degrees = [d for d in degrees if d != 0]
        if len(degrees) == 0:
            raise ValueError(
                "At least one non-zero degree polynomial must be provided to "
                "generate_minimax_sign_coeffs(). "
            )

        coeffs_flat = self.backend.GenerateMinimaxSignCoeffs(
            degrees, prec, logalpha, logerr, int(debug)
        )

        coeffs_flat = torch.tensor(coeffs_flat)
        splits = [degree + 1 for degree in degrees]
        return torch.split(coeffs_flat, splits)

    def get_depth(self, poly):
        # Older backends may not expose GetPolyDepth; treat as unknown.
        getter = getattr(self.backend, "GetPolyDepth", None)
        if getter is None:
            return None
        return getter(poly)
