import random
import subprocess
import sys
from pathlib import Path

REMOTE_HOST = "gbar"
REMOTE_DIR = "/dtu/datasets1/02516/potholes"

LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "potholes"
LOCAL_IMAGES_DIR = LOCAL_DIR / "images"
LOCAL_ANNOTATIONS_DIR = LOCAL_DIR / "annotations"


def list_remote_images():
    result = subprocess.run(
        ["ssh", REMOTE_HOST, f"ls {REMOTE_DIR}/images"],
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line.startswith("potholes") and line.endswith(".png")]


def scp(remote_path, local_dir):
    subprocess.run(["scp", f"{REMOTE_HOST}:{remote_path}", str(local_dir)], check=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing images on {REMOTE_HOST}:{REMOTE_DIR}/images ...")
    all_images = list_remote_images()
    print(f"Found {len(all_images)} images, sampling {n}")

    chosen = random.sample(all_images, n)

    for image_name in chosen:
        stem = Path(image_name).stem
        xml_name = f"{stem}.xml"

        print(f"Downloading {image_name} + {xml_name}")
        scp(f"{REMOTE_DIR}/images/{image_name}", LOCAL_IMAGES_DIR)
        scp(f"{REMOTE_DIR}/annotations/{xml_name}", LOCAL_ANNOTATIONS_DIR)

    print(f"Done. Saved to {LOCAL_DIR}")


if __name__ == "__main__":
    main()