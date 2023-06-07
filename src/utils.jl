# compute binary crossentropy loss
function crossentropy(p, l)
    # add epsilon perturbation for floating-point stability
    return -(l*log(p+eps(eltype(p))) + (1-l)*log(1-p + eps(eltype(p))))
  end