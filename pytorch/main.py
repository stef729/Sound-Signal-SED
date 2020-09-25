import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.append('D:/Project/Sound/Target Detection/utils')
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, 
    load_scalar, calculate_metrics)
from data_generator import DataGenerator
from models import (Cnn_5layers_AvgPooling, Cnn_9layers_AvgPooling, 
    Cnn_9layers_MaxPooling, Cnn_13layers_AvgPooling)
from losses import event_spatial_loss
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward
import config


def train(dataset_dir, workspace):
    '''Train. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      audio_type: 'foa' | 'mic'
      holdout_fold: '1' | '2' | '3' | '4' | 'none', set to none if using all 
        data without validation to train
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arugments & parameters
    # dataset_dir = args.dataset_dir
    # workspace = args.workspace
    # audio_type = args.audio_type
    # holdout_fold = args.holdout_fold
    # model_type = args.model_type
    # batch_size = args.batch_size
    # cuda = args.cuda and torch.cuda.is_available()
    # mini_data = args.mini_data
    # filename = args.filename
    
    # Test 1
    audio_type = 'foa'
    holdout_fold = '1'
    model_type = 'Cnn_9layers_AvgPooling'
    batch_size = 32
    cuda = True
    mini_data = True
    filename = 'train'  
    
    
    
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    max_validate_num = None # Number of audio recordings to validate
    reduce_lr = True        # Reduce learning rate after several iterations
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    metadata_dir = os.path.join(dataset_dir, 'metadata_dev')
    features_dir = os.path.join(workspace, 'features')
    scalar_path = os.path.join(workspace, 'scalars', 'scalar.h5')
        
    checkpoints_dir = os.path.join(workspace, filename, 'checkpoints', model_type)
    create_folder(checkpoints_dir)
    temp_submissions_dir = os.path.join(workspace, filename, 'submissions', model_type)
    create_folder(temp_submissions_dir)
    validate_statistics_path = os.path.join(workspace, filename, 'statistics', model_type, 'validate_statistics.pickle')
    create_folder(os.path.dirname(validate_statistics_path))
    logs_dir = os.path.join(workspace, filename, 'logs', model_type)
    create_logging(logs_dir, filemode='w')
        
    if cuda:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        
    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    if cuda:
        model.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    # Data generator
    data_generator = DataGenerator(
        features_dir=features_dir, 
        scalar=scalar, 
        batch_size=batch_size, 
        holdout_fold=holdout_fold)
        
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)

    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)

    train_bgn_time = time.time()
    iteration = 0

    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
                 
        # Evaluate
        if iteration % 200 == 0:

            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()
            

            # Uncomment for evaluating on training dataset
            train_statistics = evaluator.evaluate(
                data_type='train', 
                metadata_dir=metadata_dir, 
                submissions_dir=temp_submissions_dir, 
                max_validate_num=max_validate_num)

            
            if holdout_fold != 'none':
                validate_statistics = evaluator.evaluate(
                    data_type='validate', 
                    metadata_dir=metadata_dir, 
                    submissions_dir=temp_submissions_dir, 
                    max_validate_num=max_validate_num)
                    
                validate_statistics_container.append_and_dump(
                    iteration, validate_statistics)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)

        # Train
        model.train()
        batch_output_dict = model(batch_data_dict['feature'])
        loss = event_spatial_loss(batch_output_dict, batch_data_dict)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 1000:
            break
            
        iteration += 1


def inference_validation(dataset_dir, workspace):
    '''Inference validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      audio_type: 'foa' | 'mic'
      holdout_fold: '1' | '2' | '3' | '4' | 'none', where 'none' represents
          summary and print results of all folds 1, 2, 3 and 4. 
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      iteration: int, load model of this iteration
      batch_size: int
      cuda: bool
      visualize: bool
      mini_data: bool, set True for debugging on a small part of data
    '''

    # # Arugments & parameters
    # dataset_dir = args.dataset_dir
    # workspace = args.workspace
    # audio_type = args.audio_type
    # holdout_fold = args.holdout_fold
    # model_type = args.model_type
    # iteration = args.iteration
    # batch_size = args.batch_size
    # cuda = args.cuda and torch.cuda.is_available()
    # visualize = args.visualize
    # mini_data = args.mini_data
    # filename = args.filename
    
    
    # Test 1
    audio_type = 'foa'
    holdout_fold = '1'
    model_type = 'Cnn_9layers_AvgPooling'
    iteration = 1000
    batch_size = 32
    cuda = True
    visualize = True
    mini_data = True
    filename = 'train'  
    
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num

    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    metadata_dir = os.path.join(dataset_dir, 'metadata_dev')
    submissions_dir = os.path.join(workspace, filename, 'submissions', model_type, 'iteration={}'.format(iteration))
    create_folder(submissions_dir)    
    logs_dir = os.path.join(workspace, filename, 'logs', model_type)
    create_logging(logs_dir, filemode='w')    
    

    # Inference and calculate metrics for a fold    
    if holdout_fold != 'none':
        
        features_dir = os.path.join(workspace, 'features')
        scalar_path = os.path.join(workspace, 'scalars', 'scalar.h5')
        checkoutpoint_path = os.path.join(workspace, filename, 'checkpoints', model_type, '{}_iterations.pth'.format(iteration))
    
        # Load scalar
        scalar = load_scalar(scalar_path)
        
        # Load model    
        Model = eval(model_type)
        model = Model(classes_num)
        checkpoint = torch.load(checkoutpoint_path)
        model.load_state_dict(checkpoint['model'])
        
        if cuda:
            model.cuda()
            
        # Data generator
        data_generator = DataGenerator(
            features_dir=features_dir, 
            scalar=scalar, 
            batch_size=batch_size, 
            holdout_fold=holdout_fold)
            
        # Evaluator
        evaluator = Evaluator(
            model=model, 
            data_generator=data_generator, 
            cuda=cuda)
        
        # Calculate metrics
        data_type = 'validate'
        
        evaluator.evaluate(
            data_type=data_type, 
            metadata_dir=metadata_dir, 
            submissions_dir=submissions_dir, 
            max_validate_num=None)
        
        # Visualize reference and predicted events, elevation and azimuth
        if visualize:
            evaluator.visualize(data_type=data_type)
            
    # Calculate metrics for all 4 folds
    else:
        prediction_names = os.listdir(submissions_dir)
        prediction_paths = [os.path.join(submissions_dir, name) for \
            name in prediction_names]
        
        metrics = calculate_metrics(metadata_dir=metadata_dir, 
            prediction_paths=prediction_paths)
        
        logging.info('Metrics of {} files: '.format(len(prediction_names)))
        for key in metrics.keys():
            logging.info('    {:<20} {:.3f}'.format(key + ' :', metrics[key]))    
    

if __name__ == '__main__':
    
    dataset_dir = 'D:/Project/Sound/Target Detection/data/development'
    workspace = 'D:/Project/Sound/Target Detection/1_results'
    
    # if args.mode == 'train':
    if False:
        train(dataset_dir, workspace)

    # elif args.mode == 'inference_validation':
    elif True:
        inference_validation(dataset_dir, workspace)

    else:
        raise Exception('Error argument!')