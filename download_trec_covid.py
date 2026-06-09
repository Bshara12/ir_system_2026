import sys

sys.path.insert(0, ".")

from services.indexing.ir_datasets_adapter import IrDatasetsAdapter


adapter = IrDatasetsAdapter()

print("Downloading trec-covid...")
print("This may take several minutes depending on your internet speed.")

paths = adapter.save_to_jsonl(
    "trec-covid",
    output_dir="data/datasets/trec-covid",
    save_queries=True,
    save_qrels=True,
)

print("Download completed!")
print("Saved files:")
for key, value in paths.items():
    print(f"  {key}: {value}")