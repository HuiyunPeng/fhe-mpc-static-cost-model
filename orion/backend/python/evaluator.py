class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend
        self.new_evaluator()

    def new_evaluator(self):
        self.backend.NewEvaluator()
    
    def _log_primitive(self, primitive, **params):
        tracer = getattr(self.scheme, "trace_logger", None)
        if tracer:
            tracer.log_primitive(primitive, params)

    def add_rotation_key(self, amount: int):
        self.backend.AddRotationKey(amount)

    def negate(self, ctxt):
        self._log_primitive("Negate")
        return self.backend.Negate(ctxt)
    
    def rotate(self, ctxt, amount, in_place):
        self._log_primitive("HRot", amount=amount, in_place=bool(in_place))
        if in_place:
            return self.backend.Rotate(ctxt, amount)
        return self.backend.RotateNew(ctxt, amount)

    def add_scalar(self, ctxt, scalar, in_place):
        self._log_primitive("HAdd", kind="scalar", in_place=bool(in_place))
        if in_place:
            return self.backend.AddScalar(ctxt, float(scalar))
        return self.backend.AddScalarNew(ctxt, float(scalar))

    def sub_scalar(self, ctxt, scalar, in_place):
        self._log_primitive("HAdd", kind="scalar_sub", in_place=bool(in_place))
        if in_place:
            return self.backend.SubScalar(ctxt, float(scalar))
        return self.backend.SubScalarNew(ctxt, float(scalar))

    def mul_scalar(self, ctxt, scalar, in_place):
        scalar_type = "int" if isinstance(scalar, int) else "float"
        self._log_primitive("PMult", kind="scalar", scalar_type=scalar_type,
                            in_place=bool(in_place))
        if isinstance(scalar, float) and scalar.is_integer():
            scalar = int(scalar)  # (e.g., 1.00 -> 1)

        if isinstance(scalar, int):
            ct_out = (self.backend.MulScalarInt if in_place 
                      else self.backend.MulScalarIntNew)(ctxt, scalar)
        else:
            ct_out = (self.backend.MulScalarFloat if in_place 
                      else self.backend.MulScalarFloatNew)(ctxt, scalar)
            self._log_primitive("Rescale", origin="mul_scalar")
            ct_out = self.backend.Rescale(ct_out)

        return ct_out
        
    def add_plaintext(self, ctxt, ptxt, in_place):
        self._log_primitive("HAdd", kind="plaintext", in_place=bool(in_place))
        if in_place:
            return self.backend.AddPlaintext(ctxt, ptxt) 
        return self.backend.AddPlaintextNew(ctxt, ptxt) 

    def sub_plaintext(self, ctxt, ptxt, in_place):
        self._log_primitive("HAdd", kind="plaintext_sub", in_place=bool(in_place))
        if in_place:
            return self.backend.SubPlaintext(ctxt, ptxt) 
        return self.backend.SubPlaintextNew(ctxt, ptxt) 

    def mul_plaintext(self, ctxt, ptxt, in_place):
        self._log_primitive("PMult", kind="plaintext", in_place=bool(in_place))
        if in_place: # ct_out = ctxt
            ct_out = self.backend.MulPlaintext(ctxt, ptxt)
        else:
            ct_out = self.backend.MulPlaintextNew(ctxt, ptxt) 
        
        self._log_primitive("Rescale", origin="mul_plaintext")
        return self.backend.Rescale(ct_out)

    def add_ciphertext(self, ctxt0, ctxt1, in_place):
        self._log_primitive("HAdd", kind="ciphertext", in_place=bool(in_place))
        if in_place:
            return self.backend.AddCiphertext(ctxt0, ctxt1)
        return self.backend.AddCiphertextNew(ctxt0, ctxt1)

    def sub_ciphertext(self, ctxt0, ctxt1, in_place):
        self._log_primitive("HAdd", kind="ciphertext_sub", in_place=bool(in_place))
        if in_place:
            return self.backend.SubCiphertext(ctxt0, ctxt1)
        return self.backend.SubCiphertextNew(ctxt0, ctxt1)

    def mul_ciphertext(self, ctxt0, ctxt1, in_place):
        self._log_primitive("HMult", in_place=bool(in_place))
        if in_place: # ct_out = ctxt
            ct_out = self.backend.MulRelinCiphertext(ctxt0, ctxt1)
        else:
            ct_out = self.backend.MulRelinCiphertextNew(ctxt0, ctxt1)
        
        self._log_primitive("Rescale", origin="mul_ciphertext")
        return self.backend.Rescale(ct_out)
    
    def rescale(self, ctxt, in_place):
        self._log_primitive("Rescale", in_place=bool(in_place))
        if in_place:
            return self.backend.Rescale(ctxt)
        return self.backend.RescaleNew(ctxt)
    
    def get_live_plaintexts(self):
        return self.backend.GetLivePlaintexts() 

    def get_live_ciphertexts(self):
        return self.backend.GetLiveCiphertexts() 
