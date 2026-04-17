from sentence_transformers import SentenceTransformer, export_optimized_onnx_model
from pathlib import Path

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
onnx_dir = Path("onnx_model")

print("Loading model with ONNX backend...")
model = SentenceTransformer(model_name, backend="onnx")
model.save_pretrained(onnx_dir)
print("Export complete. Files saved in ./onnx_model")

print("Applying O3 graph optimization...")
export_optimized_onnx_model(
    model,
    optimization_config="O3",
    model_name_or_path=str(onnx_dir),
)
print("Optimization complete. Optimized model saved in ./onnx_model")
