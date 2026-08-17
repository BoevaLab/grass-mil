from grass_mil._compat import allow_omegaconf_in_checkpoints

# Applied at import so every entrypoint and any user code that loads a
# checkpoint gets it. See _compat for why this is needed under torch>=2.6.
allow_omegaconf_in_checkpoints()
