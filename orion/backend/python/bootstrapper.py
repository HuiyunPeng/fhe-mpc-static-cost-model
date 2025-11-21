class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend
    
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
        self._log_primitive("Bootstrap", slots=slots)
        return self.backend.Bootstrap(ctxt, slots)
