class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend

    def _ciphertext_level(self, ctxt):
        try:
            return int(self.backend.GetCiphertextLevel(ctxt))
        except Exception:
            return None

    def _log_primitive(self, primitive, **params):
        tracer = getattr(self.scheme, "trace_logger", None)
        if tracer:
            tracer.log_primitive(primitive, params)

    def __del__(self):
        self.backend.DeleteBootstrappers()

    def generate_bootstrapper(self, slots):
        # We will wait to instantiate any bootstrapper until our bootstrap
        # placement algorithm determines they're necessary.
        logp = self.scheme.params.get_boot_logp()
        return self.backend.NewBootstrapper(logp, slots)
    
    def bootstrap(self, ctxt, slots):
        input_level = self._ciphertext_level(ctxt)
        ct_out = self.backend.Bootstrap(ctxt, slots)
        self._log_primitive(
            "Bootstrap",
            slots=slots,
            level=self._ciphertext_level(ct_out),
            input_level=input_level,
        )
        return ct_out
