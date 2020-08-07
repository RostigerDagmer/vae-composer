import os
from pathlib import Path
import torch
from datasets.collection import BigMIDISet
from datasets.MidiFile import MidiFileF
from torch.utils.data import DataLoader
from models.ComposerVAE import ComposerVAE
import numpy as np
from utils.logging import *

def store_dataset_plausible(answer="no"):
    cwd = Path(os.getcwd())
    dataset_dir = cwd.parent / "datasets" / "midiset"
    dataset = BigMIDISet(rootdir=dataset_dir)

    t = dataset[0]
    dataset_lower_bound, size_name = convert_size(t.element_size() * t.nelement() * len(dataset))
    tm = tensor_memory(t, return_val=True)
    print(f"samples in dataset: { len(dataset) }")
    print(f"shape of sample[0]: { t.shape }") # because
    print(f"tensor memory of sample[0]: { tm[0], tm[1] }")
    print(f"dataset memory lower bound: { dataset_lower_bound, size_name }") # ~> 1.24TB this is why saving the dataset to disk is not efficient
    # instead we use a streaming Dataloader that uses GPU tensor operations to build the dataset samples into VRAM on demand

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # batch has dims B x T x N * 96 x 128
    batch = batch.reshape(batch.shape[0], batch, song.shape[1] // 96, 96, 128)
    torch.cat
    return torch.utils.data.dataloader.default_collate(batch)

def test_more_tracks():
    cwd = Path(os.getcwd())
    dataset_dir = cwd.parent / "notebooks" / "quake2.mid"
    midi = MidiFileF(dataset_dir)
    roll = midi.roll_as_tensor() # comes out as t  x n*96 x 128
    roll = roll.reshape(roll.shape[0], roll.shape[1] // 96, 96, 128)
    roll2 = roll[:,0:16,:,:].clone()
    #roll2 = roll[:, 0:16*96, :].clone()
    rolls = [roll, roll2]
    print(roll.shape)
    print(torch.cat(rolls, dim=1).shape)

    view_pianoroll(roll.sum(dim=0)[31])
    print(len(midi.tracks))

def test_model():
    input_list = []
    cwd = Path(os.getcwd())
    dataset_dir = cwd.parent / "datasets" / "midiset"
    dataset = BigMIDISet(rootdir=dataset_dir)

    # get single sample batch from dataset
    dl = DataLoader(dataset, collate_fn=collate_fn, batch_size=32, shuffle=True)

    model = ComposerVAE()
    sample_batch = next(iter(dl))
    model(sample_batch)

def main():
    test_more_tracks()


if __name__ == '__main__':
  main()


