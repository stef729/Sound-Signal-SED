import torch


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x
    
    
def interpolate(x, ratio):
    '''Interpolate the prediction to have the same time_steps as the target. 
    The time_steps mismatch is caused by maxpooling in CNN. 
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to upsample
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled
    
    
def forward(model, generate_func, cuda, return_input=False, 
    return_target=False):
    '''Forward data to model in mini-batch. 
    
    Args: 
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool

    Returns:
      list_dict, e.g.:
        [{'name': 'split1_ir0_ov1_7', 
          'output_event': (1, frames_num, classes_num), 
          'output_elevation': (1, frames_num, classes_num), 
          'output_azimuth': (1, frames_num, classes_num), 
          ...
          }, 
          ...]
    '''

    list_dict = []
    
    # Evaluate on mini-batch
    for (n, single_data_dict) in enumerate(generate_func):
        
        # Predict
        batch_feature = move_data_to_gpu(single_data_dict['feature'], cuda)
        
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_feature)

        output_dict = {
            'name': single_data_dict['name'], 
            'output_event': batch_output_dict['event'].data.cpu().numpy()}
            
        if return_input:
            output_dict['feature'] = single_data_dict['feature']
            
        if return_target:
            output_dict['target_event'] = single_data_dict['event']
            
        list_dict.append(output_dict)
        
    return list_dict