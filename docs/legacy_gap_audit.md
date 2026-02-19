# Legacy Gap Audit

This project intentionally does not copy legacy implementation details directly.
Instead, it ports validated concepts into a clean contract-first architecture.

## Resolved Legacy Risks

1. Broken module registration patterns

- Risk: modules stored in plain Python lists are not registered in `nn.Module`.
- Resolution: always use `nn.ModuleList`/`nn.ModuleDict` for learnable submodules.

2. Fragile attention assumptions

- Risk: implicit transpose/shape assumptions lead to silent logic errors.
- Resolution: explicit input-output shape contracts and attention normalization controlled by caller.

3. Hidden mutable dataset/sampler coupling

- Risk: mutating dataset-global indices/cache from training code introduces race conditions and stale state.
- Resolution: sampler strategies operate on provided graph units only, no mutation of datamodule internals.

4. Hardcoded script behavior

- Risk: project, path, split, and task logic hardcoded in scripts prevents reuse.
- Resolution: all component behavior and toggles are Hydra-configurable.

5. Over-customized message passing

- Risk: custom conv implementations increase maintenance burden and bug surface.
- Resolution: use native PyG layers by default; add only thin adapters when strictly needed.

6. Inconsistent loss interfaces

- Risk: different tasks using incompatible shape conventions.
- Resolution: standardized loss interfaces with validation for categorical, regression, and Cox objectives.

7. Optional feature leakage

- Risk: attention/SSL paths accidentally active by default due to implicit construction.
- Resolution: explicit config flags (`use_attention`, `use_ssl`) govern instantiation.

8. LOOCV split leakage in legacy scripts

- Risk: legacy LOOCV scripts computed class weights and dataset-wide metadata before or across fold boundaries, allowing held-out fold information to influence training.
- Resolution: configurable LOOCV split assignment is performed in precompute, reducer `train_only` fitting is guarded for fold-specific preprocessing, and validation strategy is explicit (`heldout_fold_items` or `patches_from_train_items`) with deterministic seeds.

## Acceptance Criteria

- Component modules are side-effect free.
- Sampler API is data-layer agnostic.
- Optional modules instantiate only when enabled.
- Tests cover shape contracts and native/custom sampler parity on synthetic inputs.
