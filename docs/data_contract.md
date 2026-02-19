# Data Contract

This document describes every file currently under `data/`, based on direct file inspection.

- Inspection date: February 17, 2026
- Root: `data/`
- Rule used: infer only from actual file contents (no guessed columns)

## Directory Layout

```text
data/
  ExtendedKG/
    augmentation_stats.json
    disease_pathway_direct_mapped.csv
    disease_protein_data_subset.csv
    indication_data_subset.csv
    kg_filtered_subset_ext_drugprotein.csv
    nodes.csv
    pathway_pathway_data_subset.csv
  External_Full/
    transductive_0.5.csv
    transductive_1.0.csv
    transductive_2.0.csv
    transductive_3.0.csv
    transductive_4.0.csv
  HighOrder/
    HO.csv
```

## ExtendedKG

### `data/ExtendedKG/augmentation_stats.json`
- Format: JSON object (`utf-8`)
- Keys (10):  
  `quads_total`, `quads_used`, `require_indication`, `indication_pairs`, `existing_drug_protein`, `existing_protein_pathway`, `existing_disease_pathway`, `added_drug_protein`, `added_protein_pathway`, `added_disease_pathway`
- Missing: n/a for JSON scalar keys (all keys present)
- Sample:

```json
{
  "quads_total": 412,
  "quads_used": 412,
  "require_indication": false,
  "indication_pairs": 0,
  "existing_drug_protein": 342501
}
```

### `data/ExtendedKG/disease_pathway_direct_mapped.csv`
- Format: CSV (`utf-8`)
- Columns: `disease_id`, `pathway_id`
- Rows: `233870`
- Missing counts:
  - `disease_id`: `0`
  - `pathway_id`: `0`
- Sample rows:

```csv
disease::9916,pathway::R-HSA-193048
disease::9916,pathway::R-HSA-535734
```

### `data/ExtendedKG/disease_protein_data_subset.csv`
- Format: CSV (`utf-8`)
- Columns: `relation`, `x_id`, `x_type`, `y_id`, `y_type`
- Rows: `5973837`
- Missing counts:
  - `relation`: `0`
  - `x_id`: `0`
  - `x_type`: `0`
  - `y_id`: `0`
  - `y_type`: `0`
- Sample rows:

```csv
disease_protein,gene/protein::1,gene/protein,disease::5090_13498_8414_10897_33312_10943_11552_14092_12054_11960_11280_11294_11295_11298_11307_11498_12879_13089_13506,disease
disease_protein,gene/protein::19,gene/protein,disease::5090_13498_8414_10897_33312_10943_11552_14092_12054_11960_11280_11294_11295_11298_11307_11498_12879_13089_13506,disease
```

### `data/ExtendedKG/indication_data_subset.csv`
- Format: CSV (`utf-8`)
- Columns: `relation`, `x_id`, `x_type`, `y_id`, `y_type`
- Rows: `18776`
- Missing counts:
  - `relation`: `0`
  - `x_id`: `0`
  - `x_type`: `0`
  - `y_id`: `0`
  - `y_type`: `0`
- Sample rows:

```csv
indication,drug::DB00492,drug,disease::5044,disease
indication,drug::DB00492,drug,disease::1200_1134_15512_5080_100078,disease
```

### `data/ExtendedKG/kg_filtered_subset_ext_drugprotein.csv`
- Format: CSV (`utf-8`)
- Columns: `relation`, `x_id`, `x_type`, `y_id`, `y_type`
- Rows: `7097999`
- Missing counts:
  - `relation`: `0`
  - `x_id`: `0`
  - `x_type`: `0`
  - `y_id`: `0`
  - `y_type`: `0`
- Sample rows:

```csv
protein_protein,gene/protein::9796,gene/protein,gene/protein::56992,gene/protein
protein_protein,gene/protein::7918,gene/protein,gene/protein::9240,gene/protein
```

### `data/ExtendedKG/nodes.csv`
- Format: CSV (`utf-8`)
- Columns: `id`, `type`, `name`, `source`
- Rows: `47559`
- Missing counts:
  - `id`: `0`
  - `type`: `0`
  - `name`: `47479`
  - `source`: `47479`
- Sample rows:

```csv
disease::1,disease,,
disease::1000,disease,,
```

### `data/ExtendedKG/pathway_pathway_data_subset.csv`
- Format: CSV (`utf-8`)
- Columns: `relation`, `x_id`, `x_type`, `y_id`, `y_type`
- Rows: `5070`
- Missing counts:
  - `relation`: `0`
  - `x_id`: `0`
  - `x_type`: `0`
  - `y_id`: `0`
  - `y_type`: `0`
- Sample rows:

```csv
pathway_pathway,pathway::R-HSA-109581,pathway,pathway::R-HSA-109606,pathway
pathway_pathway,pathway::R-HSA-109581,pathway,pathway::R-HSA-169911,pathway
```

## External_Full

All files in this directory share the same schema and encoding.

- Format: CSV (`utf-8`)
- Columns: `DrugBank_ID`, `diseaseId`, `targetId`, `phase`
- Missing counts in every file: all columns `0`

### `data/External_Full/transductive_0.5.csv`
- Rows: `153`
- Sample rows:

```csv
drug::DB00169,disease::13662,7421,0.5
drug::DB00338,disease::13662,495,0.5
```

### `data/External_Full/transductive_1.0.csv`
- Rows: `1376`
- Sample rows:

```csv
drug::DB00020,disease::4970,1438,1.0
drug::DB01611,disease::4970,51284,1.0
```

### `data/External_Full/transductive_2.0.csv`
- Rows: `2387`
- Sample rows:

```csv
drug::DB08899,disease::4970,367,2.0
drug::DB00072,disease::4970,2064,2.0
```

### `data/External_Full/transductive_3.0.csv`
- Rows: `1367`
- Sample rows:

```csv
drug::DB00112,disease::4970,7422,3.0
drug::DB09257,disease::4970,1806,3.0
```

### `data/External_Full/transductive_4.0.csv`
- Rows: `338`
- Sample rows:

```csv
drug::DB00385,disease::4993,7153,4.0
drug::DB01128,disease::4993,367,4.0
```

## HighOrder

### `data/HighOrder/HO.csv`
- Format: CSV (decoded with `cp1252` fallback; `utf-8` strict decode fails)
- Columns: `drugbank_id`, `evidence_level`, `repurposing_potential`, `rationale`, `support_pmids`, `mechanism_summary`, `support_pmids_2`, `protein_id`, `pathway_id`, `disease_id`
- Rows: `770`
- Missing counts:
  - `drugbank_id`: `0`
  - `evidence_level`: `0`
  - `repurposing_potential`: `0`
  - `rationale`: `0`
  - `support_pmids`: `764`
  - `mechanism_summary`: `439`
  - `support_pmids_2`: `763`
  - `protein_id`: `0`
  - `pathway_id`: `1`
  - `disease_id`: `1`
- Sample rows (first row shown with long text truncated for readability):

```csv
DB00334,Moderate,Low_potential,"The provided abstracts support a role for muscarinic acetylcholine receptor signaling in mood disorders ...",,,,gene/protein::1129,http://bioregistry.io/reactome:R-HSA-390648,disease::4985_693_24613_1866
DB00252,Moderate,Medium_potential,"Across the provided abstracts, SCN8A encodes the Nav1.6 voltage-gated sodium channel ...",,,,gene/protein::6334,http://bioregistry.io/reactome:R-HSA-5576892,disease::2125
```
