# Niche analysis

Niches are recurrent cellular neighbourhoods that a trained model attends to.
The analysis runs on the per-cell outputs of a model and describes every niche
by what it contains, how it relates to the phenotype, how much the model relies
on it, and how its cells are arranged.

## 1. Per-cell outputs

```bash
grass-mil-extract experiment=my_cohort extract.nested_cv_summary=<nested_cv>/nested_cv_summary.json
```

Every model of the cross-validation (`fold0`, `fold1`, ...) scores every cell of
every unit. For each cell the cache stores the encoder embedding, the attention
head's low-dimensional pre-tanh activation (the *attention space*), the
instance logits and the attention scores, with the cell's type, position and
id. The units each model held out are recorded. Explicit checkpoints can be
added with `+extract.models.<name>=<checkpoint>`.

## 2. Niches of one model

```bash
grass-mil-niches cells_dir=<extract>/cells/fold0
```

**Cells.** By default the units the model did not see during training
(`units=held_out`; or `all`, or a list).

**Margin.** Within a unit, a cell's contribution to class `c` is
`A[i, c] (l[i, c] - r_c)`, where `r_c` is the model's mean instance logit over
every cell of the cohort (recorded by `grass-mil-extract`); summed over the unit
it gives the unit's logit minus `r_c`. A positive contribution means the cell
pushes its unit above what an average cell would. (The head bias is not used as
the reference: the embeddings are not centred, so `l - b` carries a
model-specific offset that would set the sign of every margin.) A cell's margin
is its contribution to the positive class minus its contribution to the other
class(es), multiplied by the unit's number of cells so that units of different
size are comparable.

**Clustering.** Cells are clustered in the attention space
(`clustering.space=attention`; `embedding` uses the encoder embedding, for which
`clustering.pca_components=10` is recommended): L2 normalisation (cosine
geometry), HDBSCAN (`min_cluster_size` 300, `min_samples` 1, excess of mass) on
a sample of at most `clustering.sample_size` cells, and assignment of the rest
by their 15 nearest neighbours. Cells HDBSCAN leaves unassigned form the
**Background**. Niches are numbered by their margin: `Niche 1` is the most
positive.

### Outputs

| File | Content |
|---|---|
| `niches.csv` | one row per niche (see below) |
| `unit_niche_shares.csv` | the share of every niche in every unit |
| `cells.csv` | every cell: unit, id, position, cell type, niche, margin |
| `descriptors/*.csv` | spatial descriptors of every niche (see below) |
| `niche_colocalisation.csv` | `log2` observed/expected contacts between niches |
| `report.md` | the niche table and the main descriptors |
| `figures/` | see below |
| `niche_model.joblib` | the fitted clustering (labels new cells of the same model) |

`niches.csv` columns:

* `share`: fraction of the analysed cells in the niche; `n_units`: units where
  it exceeds `present_threshold` (2%);
* `association`: fraction of the niche's cells that come from positive units
  (cell-weighted; compare with the base rate in the report). For survival
  models a positive unit is one with an event;
* `margin` [`margin_lo`, `margin_hi`]: the mean over units of the niche's mean
  cell margin in the unit, with a 95% bootstrap interval that resamples
  patients (units of one patient are not independent; no interval with a
  single patient);
  `sign_consistency`: the fraction of units where it has that sign;
* `composition`, `enriched` (cell types over-represented more than
  `enrichment_threshold` times) and `frac_<cell type>`.

### Spatial descriptors

Computed on every niche's subgraph within each unit (its cells and the edges
between them) and averaged over the units where the niche has at least
`descriptors.min_cells` cells:

| File | Statistic |
|---|---|
| `moran.csv` | Moran's I of every cell type (are cells of the type clustered within the niche?), mean and s.e.m. over units, and a z-test of the mean against its expectation under random labelling (exact moments in every unit), with BH q-values over niches and types |
| `enrichment.csv` | neighbourhood enrichment of every cell-type pair against label permutations over the fixed graph: the z-score (which grows with the number of edges) and the size-free `log2` observed/expected ratio |
| `cross_l.csv` | Ripley's cross-L of every cell-type pair relative to random labelling of the niche's cells, `L_ab(r*) - L(r*)` (positive: `b` lies closer to `a` than cells in general do), at the radius `r*` where the pairs deviate most |
| `contacts.csv` | `log2` of a pair's share of the niche's contacts over its share of all contacts; masked below `descriptors.min_contacts`. It reflects the niche's composition as well as its arrangement: a pair of the niche's main cell type is enriched by composition alone |

### Figures

* `niche_panel.png`: composition, association, margin with its interval and share per niche;
* `embedding.png`: the clustering space (first two principal components) coloured by niche, margin and empirical association;
* `unit_composition.png`: the niche shares of every unit, by class;
* `units/*.png`: the most positive and most negative units in the tissue (cell types, margins, niches);
* `moran.png`, `enrichment.png`, `cross_l.png`, `contacts.png`, `niche_colocalisation.png`.

### Survival

With `survival.file` (one row per patient with `time` and `event` columns), the
niche shares of each patient (the mean over the patient's units) are related
to survival: a Cox hazard ratio per standard deviation of the log share of
every niche (`survival_cox.csv`), and with `survival.score=["Niche 1", "-Niche 4"]`
Kaplan-Meier curves of the score's tertiles with a log-rank test
(`figures/survival.png`). A score chosen after looking at the Cox table of the
same patients is tested on the data that selected it; its log-rank p-value is
then descriptive.

## 3. Replication across models (optional)

```bash
grass-mil-metaclusters cells_dir=<extract>/cells
```

Each model of the cross-validation has its own embedding, so its niches are
discovered separately. This step asks which niches recur:

1. `cluster`: the niches of every model on the units it held out;
2. `match`: niches of different models are grouped into metaclusters by the
   Hellinger distance of their compositions, with average linkage that never
   merges two niches of the same model; the threshold maximises the number of
   metaclusters found by at least `match.min_models` models. Outputs
   `metaclusters.csv`, the metacluster shares of every unit (labelled by the
   model that held it out) and patient-level tests of those shares;
3. `transfer`: every model labels every unit, giving training versus held-out
   behaviour, the agreement between models and consensus shares (`transfer/`;
   descriptive, since most models trained on most units);
4. `basis`: one k-means partition of a single model's space, attributed by
   every model (`basis/`).

## 4. Model-free spatial statistics (optional)

```bash
grass-mil-spatial-stats cells_dir=<extract>/cells/fold0 reference=[tumor]
```

Classical statistics of every unit compared between positive and other
patients (Mann-Whitney on patient means with AUROC, BH-corrected):
neighbourhood enrichment of every cell-type pair (`log2` observed/expected, so
units of different size are comparable), the shares of cellular neighbourhoods
(k-means of every cell's 10-nearest-neighbour composition, CLR-transformed),
contact fractions and cross-L relative to random labelling at fixed radii
between the `reference` group(s) and the others. `group_map` merges fine cell
types into groups first. `compare_by=unit` compares units instead of patients;
units of one patient are then treated as independent, so use it only when
every patient has one unit.
