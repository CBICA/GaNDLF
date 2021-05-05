import os
import ast 
import sys
import numpy as np
import yaml
import pkg_resources

def parse_version(version_string):
  '''
  Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison
  '''
  version_string_split = version_string.split('.')
  if len(version_string_split) > 3:
    del version_string_split[-1]
  return int(''.join(version_string_split))

def initialize_key(parameters, key):
  '''
  This function will initialize the key in the parameters dict to 'None' if it is absent or length is zero
  '''
  if key in parameters: 
    if len(parameters[key]) == 0: # if key is present but not defined
      parameters[key] = None
  else:
    parameters[key] = None # if key is absent

  return parameters

def parseConfig(config_file_path, version_check = True):
  '''
  This function parses the configuration file and returns a dictionary of parameters
  '''
  with open(config_file_path) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
  
  if version_check: # this is only to be used for testing
    if not('version' in params):
      sys.exit('The \'version\' key needs to be defined in config with \'minimum\' and \'maximum\' fields to determine the compatibility of configuration with code base')
    else:
      gandlf_version = pkg_resources.require('GANDLF')[0].version
      gandlf_version_int = parse_version(gandlf_version)
      min = parse_version(params['version']['minimum'])
      max = parse_version(params['version']['maximum'])
      if (min > gandlf_version_int) or (max < gandlf_version_int):
        sys.exit('Incompatible version of GANDLF detected (' + gandlf_version + ')')
  
  if ('psize' in params):
    print('WARNING: \'psize\' has been deprecated in favor of \'patch_size\'', file = sys.stderr)
    if not('patch_size' in params):
      params['patch_size'] = params['psize']

  if 'patch_size' in params:
    if len(params['patch_size']) == 2: # 2d check
        params['patch_size'].append(1) # ensuring same size during torchio processing
  else:
    sys.exit('The \'patch_size\' parameter needs to be present in the configuration file')
  
  if not('patch_sampler' in params):
    params['patch_sampler'] = 'label'

  if 'resize' in params:
    print('WARNING: \'resize\' should be defined under \'data_processing\', this will be skipped', file = sys.stderr)

  # Extrating the training parameters from the dictionary
  if 'num_epochs' in params:
    num_epochs = int(params['num_epochs'])
  else:
    num_epochs = 100
    print('Using default num_epochs: ', num_epochs)
  params['num_epochs'] = num_epochs
  
  if 'patience' in params:
    patience = int(params['patience'])
  else:
    print("Patience not given, train for full number of epochs")
    patience = num_epochs
  params['patience'] = patience

  if 'batch_size' in params:
    batch_size = int(params['batch_size'])
  else:
    batch_size = 1
    print('Using default batch_size: ', batch_size)
  params['batch_size'] = batch_size
  
  if 'amp' in params:
    print("Please define \'amp\' under \'model\'")

  if 'learning_rate' in params:
    learning_rate = float(params['learning_rate'])
  else:
    learning_rate = 0.001
    print('Using default learning_rate: ', learning_rate)
  params['learning_rate'] = learning_rate

  if 'modality' in params:
    modality = str(params['modality'])
    if modality.lower() == 'rad':
      pass
    elif modality.lower() == 'path':
      pass
    else:
      sys.exit('The \'modality\' should be set to either \'rad\' or \'path\'. Please check for spelling errors and it should be set to either of the two given options.')

  if 'loss_function' in params:
    defineDefaultLoss = False
    # check if user has passed a dict 
    if isinstance(params['loss_function'], dict): # if this is a dict
      if len(params['loss_function']) > 0: # only proceed if something is defined
        for key in params['loss_function']: # iterate through all keys
          if key == 'mse':
            if (params['loss_function'][key] == None) or not('reduction' in params['loss_function'][key]):
              params['loss_function'][key] = {}
              params['loss_function'][key]['reduction'] = 'mean'
          else:
            params['loss_function'] = key # use simple string for other functions - can be extended with parameters, if needed
      else:
        defineDefaultLoss = True
    else:      
      # check if user has passed a single string
      if params['loss_function'] == 'mse':
        params['loss_function'] = {}
        params['loss_function']['mse'] = {}
        params['loss_function']['mse']['reduction'] = 'mean'
  else:
    defineDefaultLoss = True
  if defineDefaultLoss == True:
    loss_function = 'dc'
    print('Using default loss_function: ', loss_function)
  else:
    loss_function = params['loss_function']
  params['loss_function'] = loss_function

  if 'metrics' in params:
    if 'accuracy' in params['metrics']:      
      if 'threshold' in params['metrics']['accuracy']:
        pass
      else:
        params['metrics']['accuracy']['threshold'] = 0.5
      
  else:
    sys.exit('The key \'metrics\' needs to be defined')

  if 'opt' in params:
    opt = str(params['opt'])
  else:
    opt = 'adam'
    print('Using default opt: ', opt)
  params['opt'] = opt
  
  # this is NOT a required parameter - a user should be able to train with NO augmentations
  params = initialize_key(params, 'data_augmentation')
  if not(params['data_augmentation'] == None):
    if len(params['data_augmentation']) > 0: # only when augmentations are defined
      
      # special case for spatial augmentation, which is now deprecated 
      if 'spatial' in params['data_augmentation']:
          if not('affine' in params['data_augmentation']) or not('elastic' in params['data_augmentation']):
              print('WARNING: \'spatial\' is now deprecated in favor of split \'affine\' and/or \'elastic\'', file = sys.stderr)
              params['data_augmentation']['affine'] = {}
              params['data_augmentation']['elastic'] = {}
              del params['data_augmentation']['spatial']
      
      # special case for random swapping - which takes a patch size to swap pixels around
      if 'swap' in params['data_augmentation']:
          if not(isinstance(params['data_augmentation']['swap'], dict)):
              params['data_augmentation']['swap'] = {}
          if not('patch_size' in params['data_augmentation']['swap']):
              params['data_augmentation']['swap']['patch_size'] = 15 # default
      
      # special case for random blur/noise - which takes a std-dev range
      for std_aug in ['blur', 'noise']:
        if std_aug in params['data_augmentation']:
            if not(isinstance(params['data_augmentation'][std_aug], dict)):
                params['data_augmentation'][std_aug] = {}
            if not('std' in params['data_augmentation'][std_aug]):
                params['data_augmentation'][std_aug]['std'] = [0, 1] # default

      # special case for random noise - which takes a mean range
      if 'noise' in params['data_augmentation']:
          if not(isinstance(params['data_augmentation']['noise'], dict)):
              params['data_augmentation']['noise'] = {}
          if not('mean' in params['data_augmentation']['noise']):
              params['data_augmentation']['noise']['mean'] = 0 # default
      
      # special case for augmentations that need axis defined 
      for axis_aug in ['flip', 'anisotropic']:
        if axis_aug in params['data_augmentation']:
            if not(isinstance(params['data_augmentation'][axis_aug], dict)):
                params['data_augmentation'][axis_aug] = {}
            if not('axis' in params['data_augmentation'][axis_aug]):
                params['data_augmentation'][axis_aug]['axis'] = [0,1,2] # default
      
      # special case for augmentations that need axis defined in 1,2,3
      for axis_aug in ['rotate_90', 'rotate_180']:
        if axis_aug in params['data_augmentation']:
            if not(isinstance(params['data_augmentation'][axis_aug], dict)):
                params['data_augmentation'][axis_aug] = {}
            if not('axis' in params['data_augmentation'][axis_aug]):
                params['data_augmentation'][axis_aug]['axis'] = [1,2,3] # default
      
      if 'anisotropic' in params['data_augmentation']: # special case for anisotropic
        if not('downsampling' in params['data_augmentation']['anisotropic']):
          default_downsampling = 1.5
        else:
          default_downsampling = params['data_augmentation']['anisotropic']['downsampling']
        
        initialize_downsampling = False
        if type(default_downsampling) is list:
          if len(default_downsampling) != 2:
            initialize_downsampling = True
            print('WARNING: \'anisotropic\' augmentation needs to be either a single number of a list of 2 numbers: https://torchio.readthedocs.io/transforms/augmentation.html?highlight=randomswap#torchio.transforms.RandomAnisotropy.', file = sys.stderr)
            default_downsampling = default_downsampling[0] # only 
        else:
          initialize_downsampling = True
        
        if initialize_downsampling:
          if default_downsampling < 1:
            print('WARNING: \'anisotropic\' augmentation needs the \'downsampling\' parameter to be greater than 1, defaulting to 1.5.', file = sys.stderr)
            default_downsampling = 1.5 
          params['data_augmentation']['anisotropic']['downsampling'] = default_downsampling # default
      
      # for all others, ensure probability is present
      default_probability = 0.5
      if 'default_probability' in params['data_augmentation']:
        default_probability = float(params['data_augmentation']['default_probability'])
      for key in params['data_augmentation']:
        if key != 'default_probability':
          if (params['data_augmentation'][key] == None) or not('probability' in params['data_augmentation'][key]): # when probability is not present for an augmentation, default to '1'
              if not isinstance(params['data_augmentation'][key], dict):
                params['data_augmentation'][key] = {}
              params['data_augmentation'][key]['probability'] = default_probability

  # this is NOT a required parameter - a user should be able to train with NO built-in pre-processing 
  params = initialize_key(params, 'data_preprocessing')
  if not(params['data_preprocessing'] == None):
    if len(params['data_preprocessing']) < 0: # perform this only when pre-processing is defined
      thresholdOrClip = False
      thresholdOrClipDict = ['threshold', 'clip'] # this can be extended, as required
      keysForWarning = ['resize'] # properties for which the user will see a warning

      # iterate through all keys
      for key in params['data_preprocessing']: # iterate through all keys
        # for threshold or clip, ensure min and max are defined
        if not thresholdOrClip:
          if (key in thresholdOrClipDict):
            thresholdOrClip = True # we only allow one of threshold or clip to occur and not both
            if not(isinstance(params['data_preprocessing'][key], dict)): # initialize if nothing is present
              params['data_preprocessing'][key] = {}
            
            # if one of the required parameters is not present, initialize with lowest/highest possible values
            # this ensures the absence of a field doesn't affect processing
            if not 'min' in params['data_preprocessing'][key]: 
              params['data_preprocessing'][key]['min'] = sys.float_info.min
            if not 'max' in params['data_preprocessing'][key]:
              params['data_preprocessing'][key]['max'] = sys.float_info.max
        else:
          sys.exit('Use only \'threshold\' or \'clip\', not both')

        # give a warning for resize
        if key in keysForWarning:
          print('WARNING: \'' + key + '\' is generally not recommended, as it changes image properties in unexpected ways.', file = sys.stderr)

  if 'modelName' in params:
    defineDefaultModel = False
    print('This option has been superceded by \'model\'', file=sys.stderr)
    which_model = str(params['modelName'])
  elif 'which_model' in params:
    defineDefaultModel = False
    print('This option has been superceded by \'model\'', file=sys.stderr)
    which_model = str(params['which_model'])
  else: # default case
    defineDefaultModel = True
  if defineDefaultModel == True:
    which_model = 'resunet'
    # print('Using default model: ', which_model)
  params['which_model'] = which_model
  
  if 'model' in params:

    if not(isinstance(params['model'], dict)):
      sys.exit('The \'model\' parameter needs to be populated as a dictionary')
    elif len(params['model']) == 0: # only proceed if something is defined
      sys.exit('The \'model\' parameter needs to be populated as a dictionary and should have all properties present')

    if 'amp' in params['model']:
      amp = params['model']['amp']
    else:
      amp = False
      print("NOT using Mixed Precision Training")
    params['model']['amp'] = amp

    if not('architecture' in params['model']):
      sys.exit('The \'model\' parameter needs \'architecture\' key to be defined')
    if not('final_layer' in params['model']):
      sys.exit('The \'model\' parameter needs \'final_layer\' key to be defined')
    if not('dimension' in params['model']):
      sys.exit('The \'model\' parameter needs \'dimension\' key to be defined, which should either 2 or 3')
    if not('base_filters' in params['model']):
      base_filters = 32
      params['model']['base_filters'] = base_filters
      print('Using default \'base_filters\' in \'model\': ', base_filters)
    if not('class_list' in params['model']): 
      params['model']['class_list'] = [] # ensure that this is initialized      
    if not('ignore_label_validation' in params['model']):
      params['model']['ignore_label_validation'] = None
    if not('batch_norm' in params['model']):
      params['model']['batch_norm'] = False

  else:
    sys.exit('The \'model\' parameter needs to be populated as a dictionary')
  
  if isinstance(params['model']['class_list'], str):
    try:
      params['model']['class_list'] = eval(params['model']['class_list'])
    except:
      if ('||' in params['model']['class_list']) or ('&&' in params['model']['class_list']):
        # special case for multi-class computation - this needs to be handled during one-hot encoding mask construction
        print('This is a special case for multi-class computation, where different labels are processed together')
        temp_classList = params['model']['class_list']
        temp_classList= temp_classList.replace('[', '') # we don't need the brackets
        temp_classList= temp_classList.replace(']', '') # we don't need the brackets
        params['model']['class_list'] = temp_classList.split(',')

  if 'kcross_validation' in params:
    sys.exit('\'kcross_validation\' is no longer used, please use \'nested_training\' instead')

  if not('nested_training' in params):
    sys.exit('The parameter \'nested_training\' needs to be defined')
  if not('testing' in params['nested_training']):
    if not('holdout' in params['nested_training']):
      kfolds = -5
      print('Using default folds for testing split: ', kfolds)
    else:
      print('WARNING: \'holdout\' should not be defined under \'nested_training\', please use \'testing\' instead;', file = sys.stderr)
      kfolds = params['nested_training']['holdout']
    params['nested_training']['testing'] = kfolds
  if not('validation' in params['nested_training']):
    kfolds = -5
    print('Using default folds for validation split: ', kfolds)
    params['nested_training']['validation'] = kfolds

  if not 'in_memory' in params:
    params['in_memory'] = False
  
  if not 'save_masks' in params:
    params['save_masks'] = False
    
  # Setting default values to the params
  if 'scheduler' in params:
      scheduler = str(params['scheduler'])
  else:
      scheduler = 'triangle'
  params['scheduler'] = scheduler.lower()

  if 'scaling_factor' in params:
      scaling_factor = params['scaling_factor']
  else:
      scaling_factor = 1
  params['scaling_factor'] = scaling_factor

  if 'q_max_length' in params:
      q_max_length = int(params['q_max_length'])
  else:
      q_max_length = 100
  params['q_max_length'] = q_max_length

  if 'q_samples_per_volume' in params:
      q_samples_per_volume = int(params['q_samples_per_volume'])
  else:
      q_samples_per_volume = 10
  params['q_samples_per_volume'] = q_samples_per_volume

  if int(params['q_max_length']) % int(params['q_samples_per_volume']) !=0:
      sys.exit('\'q_max_length\' needs to be divisible by \'q_samples_per_volume\'')

  if 'q_num_workers' in params:
      q_num_workers = int(params['q_num_workers'])
  else:
      q_num_workers = 4
  params['q_num_workers'] = q_num_workers

  q_verbose = False
  if 'q_verbose' in params:
      if params['q_verbose'] == 'True':
          q_verbose = True  
  params['q_verbose'] = q_verbose

  parallel_compute_command = ''
  if 'parallel_compute_command' in params:
      parallel_compute_command = params['parallel_compute_command']
      parallel_compute_command = parallel_compute_command.replace('\'', '')
      parallel_compute_command = parallel_compute_command.replace('\"', '')
  params['parallel_compute_command'] = parallel_compute_command

  if 'weighted_loss' in params:
    pass
  else:
    print("WARNING: NOT using weighted loss")
    params['weighted_loss'] = False 
  
  if 'verbose' in params:
    pass
  else:
    params['verbose'] = False

  return params
