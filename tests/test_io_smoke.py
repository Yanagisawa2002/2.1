from pathlib import Path

from src.data.io import JSONDocument, load_all_datasets


def test_load_all_datasets_smoke() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    datasets = load_all_datasets(data_dir)

    expected_paths = {
        "ExtendedKG/augmentation_stats.json",
        "ExtendedKG/disease_pathway_direct_mapped.csv",
        "ExtendedKG/disease_protein_data_subset.csv",
        "ExtendedKG/indication_data_subset.csv",
        "ExtendedKG/kg_filtered_subset_ext_drugprotein.csv",
        "ExtendedKG/nodes.csv",
        "ExtendedKG/pathway_pathway_data_subset.csv",
        "External_Full/transductive_0.5.csv",
        "External_Full/transductive_1.0.csv",
        "External_Full/transductive_2.0.csv",
        "External_Full/transductive_3.0.csv",
        "External_Full/transductive_4.0.csv",
        "HighOrder/HO.csv",
    }
    assert set(datasets) == expected_paths

    expected_columns = {
        "ExtendedKG/disease_pathway_direct_mapped.csv": ("disease_id", "pathway_id"),
        "ExtendedKG/disease_protein_data_subset.csv": (
            "relation",
            "x_id",
            "x_type",
            "y_id",
            "y_type",
        ),
        "ExtendedKG/indication_data_subset.csv": (
            "relation",
            "x_id",
            "x_type",
            "y_id",
            "y_type",
        ),
        "ExtendedKG/kg_filtered_subset_ext_drugprotein.csv": (
            "relation",
            "x_id",
            "x_type",
            "y_id",
            "y_type",
        ),
        "ExtendedKG/nodes.csv": ("id", "type", "name", "source"),
        "ExtendedKG/pathway_pathway_data_subset.csv": (
            "relation",
            "x_id",
            "x_type",
            "y_id",
            "y_type",
        ),
        "External_Full/transductive_0.5.csv": ("DrugBank_ID", "diseaseId", "targetId", "phase"),
        "External_Full/transductive_1.0.csv": ("DrugBank_ID", "diseaseId", "targetId", "phase"),
        "External_Full/transductive_2.0.csv": ("DrugBank_ID", "diseaseId", "targetId", "phase"),
        "External_Full/transductive_3.0.csv": ("DrugBank_ID", "diseaseId", "targetId", "phase"),
        "External_Full/transductive_4.0.csv": ("DrugBank_ID", "diseaseId", "targetId", "phase"),
        "HighOrder/HO.csv": (
            "drugbank_id",
            "evidence_level",
            "repurposing_potential",
            "rationale",
            "support_pmids",
            "mechanism_summary",
            "support_pmids_2",
            "protein_id",
            "pathway_id",
            "disease_id",
        ),
    }

    for relative_path, required in expected_columns.items():
        table = datasets[relative_path]
        assert tuple(table.columns) == required
        assert table.row_count > 0

    augmentation = datasets["ExtendedKG/augmentation_stats.json"]
    assert isinstance(augmentation, JSONDocument)
    assert {
        "quads_total",
        "quads_used",
        "require_indication",
        "indication_pairs",
        "existing_drug_protein",
        "existing_protein_pathway",
        "existing_disease_pathway",
        "added_drug_protein",
        "added_protein_pathway",
        "added_disease_pathway",
    } == set(augmentation.data.keys())

