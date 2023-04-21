import torch
from pathlib import Path
import wandb 
import argparse
from torch.utils.data import DataLoader
import datetime
from data_utils import KoreanFolk, random_test_ter
from trainer import Trainer, TripletTrainer
from loss import MarginHingeLoss
from pitch_utils import PitchDataset, PitchTriplet, ToriDataset
from model_zoo import CnnEncoder, CnnClassifier
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import pandas as pd


def make_experiment_name_with_date(config):
  current_time_in_str = datetime.datetime.now().strftime("%m%d-%H%M")
  return f'{current_time_in_str}_{config.exp}_{config.general.exp_name}'

@hydra.main(config_path='./yamls/', config_name='baseline')
def main(config: DictConfig):
  if config.general.make_log:
    wandb.init(
      project="korean-folk", 
      entity="maler", 
      name = make_experiment_name_with_date(config), 
      config = OmegaConf.to_container(config)
    )
    save_dir = Path(wandb.run.dir) / 'checkpoints'
  else:
    save_dir = Path('wandb/debug/checkpoints')


  meta = pd.read_csv(hydra.utils.get_original_cwd() + '/' + config.meta_csv_path)


  test_ter = random_test_ter(meta, config.train.seed)
  # run = wandb.init(project = 'PitchModel_test_1', reinit = True)
  # wandb.config.update(args)

  csv_dir = config.contour_dir

  if config.general.debug:
    max_size = 500
  else:
    max_size = -1

  if config.exp == 'self_supervised':
    model = CnnEncoder(config.model_params)

    train_set = PitchTriplet(meta, csv_dir, test_ter, min_length=config.train.min_length, slice_len=config.train.slice_len, split='train', max_size=max_size, frame_rate=config.train.frame_rate, use_pitch_aug=config.train.use_pitch_aug)
    test_set = PitchTriplet(meta, csv_dir, test_ter, min_length=config.train.min_length, slice_len=config.train.slice_len, split='test', max_size=max_size, frame_rate=config.train.frame_rate, use_pitch_aug=config.train.use_pitch_aug)
    loss_fn = MarginHingeLoss(margin=config.train.hinge_margin)

  elif config.exp == 'terrain_classifier':
    model = CnnClassifier(config.model_params)

    train_set = PitchDataset(meta, csv_dir, test_ter, min_length=config.train.min_length, slice_len=config.train.slice_len, split='train', max_size=max_size, frame_rate=config.train.frame_rate, use_pitch_aug=config.train.use_pitch_aug)
    test_set = PitchDataset(meta, csv_dir, test_ter, min_length=config.train.min_length, slice_len=config.train.slice_len, split='test', max_size=max_size, frame_rate=config.train.frame_rate, use_pitch_aug=config.train.use_pitch_aug)
    weight = train_set.get_label_weight()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
  else:
    raise NotImplementedError

  train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)
  valid_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  save_dir.mkdir(parents=True, exist_ok=True)  

  if config.downstream:
    tori_set = ToriDataset(meta, config.contour_dir, frame_rate=config.train.frame_rate, min_length=config.train.min_length, slice_len=config.train.slice_len)
    tori_set = None

  if config.exp == 'self_supervised':
    trainer = TripletTrainer(model=model,
                             train_loader=train_loader,
                              valid_loader=valid_loader,
                              optimizer=optimizer,
                              loss_fn=loss_fn,
                              device='cuda',
                              save_dir=save_dir,
                              num_iter_per_valid=config.train.num_iter_per_valid,
                              save_log=config.general.make_log,
                              downstream_dataset=tori_set,
                              )
      # model, 
      #                       train_loader, 
      #                       valid_loader, 
      #                       optimizer,  
      #                       loss_fn, 
      #                       'cuda', 
      #                       save_dir,
      #                       num_iter_per_valid=config.train.num_iter_per_valid, 
      #                       save_log=config.general.make_log,)
  elif config.exp == 'terrain_classifier':
    trainer = Trainer(model=model,  
                      train_loader=train_loader, 
                      valid_loader=valid_loader, 
                      optimizer=optimizer,  
                      loss_fn=loss_fn, 
                      device='cuda', 
                      save_dir=save_dir,
                      num_iter_per_valid=config.train.num_iter_per_valid, 
                      save_log=config.general.make_log,
                      downstream_dataset=tori_set,)
  else:
    raise NotImplementedError

  if config.general.make_log:
    wandb.watch(model)

  # trainer.train_by_num_epoch(30)
  trainer.train_by_num_iteration(config.train.total_iters)
  return



if __name__ == '__main__':
  
  '''
  For Audio Model

  for _ in range(20):
    run_audio_classification_model()

  '''

  main()