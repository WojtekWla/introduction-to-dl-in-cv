"""Load the extracted UCF10 frames and optional precomputed optical flow."""
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T


DEFAULT_ROOT = Path(__file__).resolve().parent / "ufc10"


def resolve_root(root_dir=None):
    root = Path(root_dir).expanduser() if root_dir is not None else DEFAULT_ROOT
    root = root.resolve()
    if not (root / "metadata").is_dir():
        raise FileNotFoundError(
            f"Dataset metadata not found in {root}. Pass the extracted dataset directory "
            "containing metadata/, frames/, and optionally videos/ and flows/."
        )
    return root


class _VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, split="train"):
        self.root_dir = resolve_root(root_dir)
        self.split = split
        self.df = pd.read_csv(self.root_dir / "metadata" / f"{split}.csv")
        if self.df.empty or self.df.video_name.duplicated().any():
            raise ValueError(f"Metadata for {split} must contain unique, nonempty video records")
        # Metadata defines a shared ordering for RGB and flow, even without AVI files.
        self.video_paths = [
            str(self.root_dir / "videos" / split / path)
            for path in self.df.video_path
        ]

    def __len__(self):
        return len(self.df)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def _asset_dir(self, idx, kind):
        relative = Path(self.df.iloc[idx].video_path).with_suffix("")
        return self.root_dir / kind / self.split / relative


class FrameImageDataset(_VideoDataset):
    def __init__(self, root_dir=None, split="train", transform=None):
        super().__init__(root_dir, split)
        self.transform = transform or T.ToTensor()
        self.frame_paths = []
        self.labels = []
        for idx in range(len(self.df)):
            directory = self._asset_dir(idx, "frames")
            paths = sorted(directory.glob("*.jpg"))
            if not paths:
                raise FileNotFoundError(f"No extracted frames found in {directory}")
            self.frame_paths.extend(paths)
            self.labels.extend([int(self.df.iloc[idx].label)] * len(paths))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        with Image.open(self.frame_paths[idx]) as image:
            frame = self.transform(image.convert("RGB"))
        return frame, self.labels[idx]


class FrameVideoDataset(_VideoDataset):
    def __init__(self, root_dir=None, split="train", transform=None, stack_frames=True):
        super().__init__(root_dir, split)
        self.transform = transform or T.ToTensor()
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

    def __getitem__(self, idx):
        frames = [self.transform(frame) for frame in self.load_frames(self._asset_dir(idx, "frames"))]
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames, int(self.df.iloc[idx].label)

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            with Image.open(Path(frames_dir) / f"frame_{i}.jpg") as image:
                frames.append(image.convert("RGB"))
        return frames


class FlowVideoDataset(_VideoDataset):
    def __init__(self, root_dir=None, split="train", n_flows=9):
        super().__init__(root_dir, split)
        if not 1 <= n_flows <= 9:
            raise ValueError("n_flows must be between 1 and 9 for the ten extracted frames")
        self.n_flows = n_flows

    def __getitem__(self, idx):
        flows = self.load_flows(self._asset_dir(idx, "flows"))
        return torch.from_numpy(np.concatenate(flows, axis=0)).float(), int(self.df.iloc[idx].label)

    def load_flows(self, flows_dir):
        """Reuse supplied flow, or lazily compute and cache Farneback flow."""
        flows_dir = Path(flows_dir)
        relative = flows_dir.relative_to(self.root_dir / "flows")
        frames_dir = self.root_dir / "frames" / relative
        flows = []
        for i in range(1, self.n_flows + 1):
            flow_file = flows_dir / f"flow_{i}_{i + 1}.npy"
            if flow_file.exists():
                flow = np.load(flow_file, allow_pickle=False)
            else:
                try:
                    import cv2
                except ImportError as exc:
                    raise ImportError("Generating missing flow requires opencv-python-headless. Run `uv sync`.") from exc
                images = []
                for frame_id in (i, i + 1):
                    with Image.open(frames_dir / f"frame_{frame_id}.jpg") as image:
                        images.append(np.asarray(image.convert("L")))
                flow = cv2.calcOpticalFlowFarneback(
                    images[0], images[1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                ).transpose(2, 0, 1).copy()
                flows_dir.mkdir(parents=True, exist_ok=True)
                # Atomic replacement also permits concurrent data-loader workers.
                with tempfile.NamedTemporaryFile(dir=flows_dir, suffix=".npy", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    try:
                        np.save(tmp, flow)
                        tmp.close()
                        os.replace(temp_path, flow_file)
                    finally:
                        temp_path.unlink(missing_ok=True)
            if flow.ndim != 3 or flow.shape[0] != 2:
                raise ValueError(f"Expected flow shaped [2, H, W] in {flow_file}, got {flow.shape}")
            flows.append(flow)
        return flows


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = FrameVideoDataset(split="val", transform=T.Compose([T.Resize((64, 64)), T.ToTensor()]))
    frames, labels = next(iter(DataLoader(dataset, batch_size=8)))
    print(frames.shape, labels.shape)
