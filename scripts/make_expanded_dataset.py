import pandas as pd
from pathlib import Path
from PIL import Image
import io, shutil

root = Path("data")
out = root / "expanded_train"

if out.exists():
    shutil.rmtree(out)

(out / "real").mkdir(parents=True)
(out / "fake").mkdir(parents=True)

# Copy the existing training images
for label in ["real", "fake"]:
    src = root / "raw" / "train" / label
    for i, f in enumerate(sorted(src.glob("*"))):
        if f.is_file():
            shutil.copy2(f, out / label / f"original_{i:03d}{f.suffix}")

# Extract 8 real + 8 fake images from the cached parquet
df = pd.read_parquet(root / "cache" / "tiny_train.parquet")
real_rows = df[df["label"] == 0].head(8)
fake_rows = df[df["label"] == 1].head(8)

for label, rows in [("real", real_rows), ("fake", fake_rows)]:
    for i, (_, row) in enumerate(rows.iterrows()):
        image_data = row["image"]
        if isinstance(image_data, dict):
            image_data = image_data["bytes"]
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.save(out / label / f"parquet_{i:03d}.jpg")

print("Expanded training dataset created:")
print("Real:", len(list((out / "real").glob("*"))))
print("Fake:", len(list((out / "fake").glob("*"))))
