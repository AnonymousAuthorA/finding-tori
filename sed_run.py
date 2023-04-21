from panns_inference import SoundEventDetection
import soundfile as sf
from pathlib import Path
import torchaudio
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm


class WaveSet:
  def __init__(self, path, sr=32000) -> None:
    self.path = Path(path)
    self.wav_list = sorted(list(self.path.rglob('*.wav')))
    self.sr = sr
    self.wav_list, self.audio_len = self.sort_wav_list_by_audio_len()
  
  def __len__(self):
    return len(self.wav_list)
  
  def __getitem__(self, idx):
    wav_path = self.wav_list[idx]
    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(0)

    if sr != self.sr:
      audio = torchaudio.functional.resample(audio, sr, self.sr)
    return audio
  
  def get_audio_length(self, idx):
    wav_path = self.wav_list[idx]
    ob = sf.SoundFile(wav_path)
    audio_len = ob.frames / ob.samplerate
    return audio_len
  
  def sort_wav_list_by_audio_len(self):
    audio_len_list = [self.get_audio_length(idx) for idx in range(len(self))]
    sorted_idx = torch.argsort(torch.tensor(audio_len_list))
    wav_list = [self.wav_list[idx] for idx in sorted_idx]
    sorted_audio_len_list = [audio_len_list[idx] for idx in sorted_idx]
    return wav_list, sorted_audio_len_list
  

def pad_collate(raw_batch):
  lens = [len(x) for x in raw_batch]
  max_len = max(lens)
  output = torch.zeros(len(raw_batch), max_len)

  for i, sample in enumerate(raw_batch):
    output[i, :len(sample)] = sample
  
  return output, lens


class CsvSaver:
  def __init__(self, save_dir, wav_fn_list, batch_size, target_labels, target_indices):
    self.save_dir = Path(save_dir)
    self.wav_fn_list = wav_fn_list
    self.batch_size = batch_size
    self.target_labels = target_labels
    self.target_labels = [x.replace(',', '_') for x in target_labels]
    self.target_indices = target_indices

  def __call__(self, x:np.ndarray, batch_idx:int):
    current_idx = batch_idx * self.batch_size
    for i in range(x.shape[0]):
      idx = current_idx + i
      wav_fn = Path(self.wav_fn_list[idx]).stem
      # save with target labels
      pred = x[i,:, self.target_indices]
      # pred = pred[:, self.target_indices]

      np.savetxt(self.save_dir / f'{wav_fn}_sed.csv', pred.T, 
                 delimiter=',', header=','.join(self.target_labels),
                 fmt='%.5f'
                 )

if __name__ == "__main__":
  sed = SoundEventDetection(checkpoint_path=None, device='cuda')
  batch_size = 4
  dir_path = Path('korean-folk/convert_wav')
  dataset = WaveSet(dir_path)
  data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_collate, pin_memory=True, num_workers=2, drop_last=False, shuffle=False)


  target_labels = ['Speech', 'Narration, monologue', 'Singing', 'Choir', 'Percussion', 'Musical instrument', 'Drum', 'Bass drum']
  target_indices = [sed.labels.index(x) for x in target_labels]
  saver = CsvSaver('./sed_output', dataset.wav_list, batch_size, target_labels, target_indices)

  for batch_idx, (audio, audio_len) in tqdm(enumerate(data_loader)):
    framewise_output = sed.inference(audio)
    saver(framewise_output, batch_idx)



