# Classical Spatial Statistics (`grass-mil-spatial-stats`)

Model-free statistics on the cell graph of every region, averaged per patient
and compared between classes with Mann-Whitney U and Benjamini-Hochberg
correction within each analysis. They run on any cell cache written by
`grass-mil-extract` (only cell types, positions and the graph are used).

```bash
grass-mil-spatial-stats cells_dir=runs/cells/fold0 output_dir=runs/spatial \
  labels.label_file=data/raw/labels.csv labels.label_column=STAS labels.patient_column=PATIENT \
  reference=[tumor] partners=[myeloid,CD8,Treg] infiltrating=[myeloid,CD8,Treg] \
  group_map='{normal: tumor, hypoxic: tumor, cd8: CD8, ki67_cd8: CD8}'
```

| analysis | per region | patient level |
|---|---|---|
| `nhood` | neighbourhood enrichment z-scores of every group pair on the cell graph, `nhood.n_perms` label permutations (`nhood.backend=squidpy` delegates to `squidpy.gr.nhood_enrichment` on the same graph) | mean z per pair, test |
| `cn` | `cn.window_k`-nearest-neighbour window composition, `MiniBatchKMeans(cn.n_cn)` fitted on `cn.sample_size` windows of all regions, share of each cellular neighbourhood | mean share, CLR-transformed, test; `cn_centres.csv` gives the CN compositions |
| `contacts` | fraction of `reference` cells touching >= 1 cell of each partner group, homotypic contact, fraction of each `infiltrating` group whose neighbours are mostly reference cells | mean, test |
| `crossk` | Ripley cross-K reference -> partner at `crossk.radii`, normalised by pi r^2 (1 = complete spatial randomness; self pairs removed for reference -> reference) | mean, test |

Outputs: `*_per_region.csv`, `nhood_tests.csv`, `cn_tests.csv`, `contact_tests.csv`,
`crossk_tests.csv`, `report_spatial.md`, `spatial_stats_summary.json`.
`group_map` collapses fine cell types into coarse groups before everything
(identity by default). Distances are in the cache's coordinate units.
