#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:45:06 2021

@author: siddhesh
"""
import os, math
import torch
import time
import torchio
import psutil
from torch.utils.data import DataLoader
from GANDLF.logger import Logger
from GANDLF.losses import fetch_loss_function
from GANDLF.metrics import fetch_metric
from GANDLF.parameterParsing import get_model, get_optimizer, get_scheduler
from GANDLF.utils import get_date_time, send_model_to_device
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame

os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"  # hides torchio citation request

# Reminder, the scaling factor should go to the metric MSE, and all should support a scaling factor, right?

def get_loss_and_metrics(ground_truth, predicted, params):
    """
    ground_truth : torch.Tensor
        The input ground truth for the corresponding image label
    predicted : torch.Tensor
        The input predicted label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    """
    loss_function = fetch_loss_function(params["loss_function"], params)
    loss = loss_function(predicted, ground_truth, params)
    metric_output = {}
    # Metrics should be a list
    for metric in params["metrics"]:
        metric_function = fetch_metric(metric)  # Write a fetch_metric
        metric_output[metric] = metric_function(predicted, ground_truth, params).cpu().data.item()
    return loss, metric_output

def step(model, image, label, params):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image : torch.Tensor
        The input image stack according to requirements
    label : torch.Tensor
        The input label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    output: torch.Tensor
        The final output of the model

    """
    if params['verbose']:
        #print('=== Memory (allocated; cached) : ', round(torch.cuda.memory_allocated()/1024**3, 1), '; ', round(torch.cuda.memory_reserved()/1024**3, 1))
        print(torch.cuda.memory_summary())
        print('|===========================================================================|\n|                             CPU Utilization                            |\n|')
        print('Load_Percent   :', psutil.cpu_percent(interval=None))
        print('MemUtil_Percent:', psutil.virtual_memory()[2])
        print('|===========================================================================|\n|')
        
    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        if len(params['value_keys']) == 0: # squeeze label for segmentation only
            label = torch.squeeze(label, -1)
    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output = model(image)
    else:
        output = model(image)
    # print("Output shape : ", output.shape, flush=True)
    # one-hot encoding of 'output' will probably be needed for segmentation
    loss, metric_output = get_loss_and_metrics(label, output, params)
    
    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)

    return loss, metric_output, output


def train_network(model, train_dataloader, optimizer, params):
    """
    Function to train a network for a single epoch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    train_dataloader : torch.DataLoader
        The dataloader for the training epoch
    optimizer : torch.optim
        Optimizer for optimizing network
    params : dict
        the parameters passed by the user yaml

    Returns
    -------
    average_epoch_train_loss : float
        Train loss for the current epoch
    average_epoch_train_metric : dict
        Train metrics for the current epoch

    """
    # Initialize a few things
    total_epoch_train_loss = 0
    total_epoch_train_metric = {}
    average_epoch_train_metric = {}
    
    for metric in params["metrics"]:
        total_epoch_train_metric[metric] = 0

    # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    if params["model"]["amp"]:
        print("Using Automatic mixed precision", flush=True)
        scaler = torch.cuda.amp.GradScaler()

    # Fetch the optimizer

    # Set the model to train
    model.train()
    for batch_idx, (subject) in enumerate(train_dataloader):
        optimizer.zero_grad()
        image = torch.cat(
            [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
        ).float().to(params["device"])
        if len(params["value_keys"]) > 0:
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            label = label.reshape(params['batch_size'], len(params['value_keys']))
        else:
            label = subject["label"][torchio.DATA]
            # one-hot encoding of 'label' will probably be needed for segmentation
        label = label.to(params["device"])
        # print("Train : ", label.shape, image.shape, flush=True)
        loss, calculated_metrics, _ = step(model, image, label, params)
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                if not torch.isnan(
                    loss
                ):  # if loss is nan, dont backprop and dont step optimizer
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        else:
            if not math.isnan(loss):
                loss.backward()
                optimizer.step()

        # Non network training related
        total_epoch_train_loss += loss.cpu().data.item()
        for metric in calculated_metrics.keys():
            total_epoch_train_metric[metric] += calculated_metrics[metric]

        # For printing information at halftime during an epoch
        if (batch_idx+1) % (len(train_dataloader) / 2) == 0:
            print("Epoch Average Train loss : ", total_epoch_train_loss / (batch_idx+1))
            for metric in params["metrics"]:
                print(
                    "Epoch Average Train " + metric + " : ",
                    total_epoch_train_metric[metric] / (batch_idx+1),
                )

    average_epoch_train_loss = total_epoch_train_loss / len(train_dataloader)
    for metric in params["metrics"]:
        average_epoch_train_metric[metric] = total_epoch_train_metric[metric] / len(
            train_dataloader
        )

    return average_epoch_train_loss, average_epoch_train_metric


def validate_network(model, valid_dataloader, scheduler, params):
    """
    Function to validate a network for a single epoch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    valid_dataloader : torch.DataLoader
        The dataloader for the validation epoch
    params : dict
        the parameters passed by the user yaml

    Returns
    -------
    average_epoch_valid_loss : float
        Validation loss for the current epoch
    average_epoch_valid_metric : dict
        Validation metrics for the current epoch

    """
    print("*" * 20)
    print("Starting validation : ")
    print("*" * 20)
    # Initialize a few things
    total_epoch_valid_loss = 0
    total_epoch_valid_metric = {}
    average_epoch_valid_metric = {}

    for metric in params["metrics"]:
        total_epoch_valid_metric[metric] = 0

    # # automatic mixed precision - https://pytorch.org/docs/stable/amp.html
    # if params["model"]["amp"]:
    #     print("Using Automatic mixed precision", flush=True)

    # Set the model to valid
    model.eval()
    for batch_idx, (subject) in enumerate(valid_dataloader):
        
        # constructing a new dict because torchio.GridSampler requires torchio.Subject, which requires torchio.Image to be present in initial dict, which the loader does not provide
        subject_dict = {}
        label_ground_truth = None
        if ('label' in subject):
            if (subject['label'] != ['NA']):
                subject_dict['label'] = torchio.Image(subject['label']['path'], type = torchio.LABEL)
                label_ground_truth = subject["label"][torchio.DATA]
                # one-hot encoding of 'label_ground_truth' will probably be needed for segmentation
        
        if len(params["value_keys"]) > 0: # for regression/classification
            label_ground_truth = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            label_ground_truth = torch.reshape(
                subject[params["value_keys"][0]], (params["batch_size"], 1)
            )
            for key in params["value_keys"]:
                subject_dict[key] = subject[key]
        
        for key in params["channel_keys"]:
            subject_dict[key] = torchio.Image(subject[key]['path'], type=torchio.INTENSITY)
            
        grid_sampler = torchio.inference.GridSampler(torchio.Subject(subject_dict), params['patch_size'])
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        
        is_segmentation = True
        output_prediction = 0 # this is used for regression/classification
        for patches_batch in patch_loader:
            image = torch.cat(
                [patches_batch[key][torchio.DATA] for key in params["channel_keys"]], dim=1
            ).float().to(params["device"])
            if len(params["value_keys"]) > 0:
                is_segmentation = False
                label = torch.cat([patches_batch[key] for key in params["value_keys"]], dim=0)
                label = torch.reshape(
                    patches_batch[params["value_keys"][0]], (params["batch_size"], 1)
                )
                # one-hot encoding of 'label' will probably be needed for segmentation
            else:
                label = patches_batch["label"][torchio.DATA]
            label = label.to(params["device"])
            # print("Validation :", label.shape, image.shape, flush=True)
            patch_location = patches_batch[torchio.LOCATION]
            loss, calculated_metrics, output = step(model, image, label, params)
            if is_segmentation:
                aggregator.add_batch(output, patch_location)
            else:
                if torch.is_tensor(output):
                    output_prediction += output.cpu().data.item()# this probably needs customization for classification (majority voting or median, perhaps?)
                else:
                    output_prediction += output
        
        if is_segmentation:
            output_prediction = aggregator.get_output_tensor()
            # reverse one-hot encoding of 'output_prediction' will probably be needed for segmentation
        else:
            output_prediction = output_prediction / len(patch_loader) # final regression output

        ## this is currently broken
        # final_loss, final_metric = get_loss_and_metrics(label_ground_truth, output_prediction, params)
        # print("Full image validation:: Loss: ", final_loss, "; Metric: ", final_metric, flush=True)

        # Non network validing related
        total_epoch_valid_loss += loss.cpu().data.item()
        for metric in calculated_metrics.keys():
            total_epoch_valid_metric[metric] += calculated_metrics[metric]

        # For printing information at halftime during an epoch
        if batch_idx != 0:
            if batch_idx % (len(valid_dataloader) // 2) == 0:
                print(
                    "Epoch Average Validation loss : ", total_epoch_valid_loss / batch_idx
                )
                for metric in params["metrics"]:
                    print(
                        "Epoch Validation " + metric + " : ",
                        total_epoch_valid_metric[metric] / len(valid_dataloader),
                    )

    average_epoch_valid_loss = total_epoch_valid_loss / len(valid_dataloader)
    for metric in params["metrics"]:
        average_epoch_valid_metric[metric] = total_epoch_valid_metric[metric] / len(
            valid_dataloader
        )
    
    if params['scheduler'] == "reduce-on-plateau":
        scheduler.step(average_epoch_valid_loss)
    else:
        scheduler.step()

    return average_epoch_valid_loss, average_epoch_valid_metric


def training_loop(
    training_data,
    validation_data,
    headers,
    device,
    params,
    output_dir,
    testing_data=None,
):

    # Some autodetermined factors
    num_classes = len(params["model"]["class_list"])
    params["headers"] = headers
    epochs = params["num_epochs"]
    loss = params["loss_function"]
    metrics = params["metrics"]
    params["device"] = device

    if not ("num_channels" in params["model"]):
        params["model"]["num_channels"] = len(headers["channelHeaders"])
    
    # ensure the number of output classes for model prediction is working correctly
    if len(headers["predictionHeaders"]) > 0:
        params["model"]["num_classes"] = len(headers["predictionHeaders"])
    else:
        params["model"]["num_classes"] = num_classes

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    # Fetch the model according to params mentioned in the configuration file
    model, params["model"]["amp"] = get_model(
        modelname=params["model"]["architecture"],
        num_dimensions=params["model"]["dimension"],
        num_channels=params["model"]["num_channels"],
        num_classes=params["model"]["num_classes"],
        base_filters=params["model"]["base_filters"],
        final_convolution_layer=params["model"]["final_layer"],
        patch_size=params["patch_size"],
        batch_size=params["batch_size"],
        amp=params["model"]["amp"]
    )

    # Set up the dataloaders
    training_data_for_torch = ImagesFromDataFrame(
        training_data,
        patch_size=params["patch_size"],
        headers=params["headers"],
        q_max_length=params["q_max_length"],
        q_samples_per_volume=params["q_samples_per_volume"],
        q_num_workers=params["q_num_workers"],
        q_verbose=params["q_verbose"],
        sampler=params["patch_sampler"],
        augmentations=params["data_augmentation"],
        preprocessing=params["data_preprocessing"],
        in_memory=params["in_memory"],
        train=True,
    )

    validation_data_for_torch = ImagesFromDataFrame(
        validation_data,
        patch_size=params["patch_size"],
        headers=params["headers"],
        q_max_length=params["q_max_length"],
        q_samples_per_volume=params["q_samples_per_volume"],
        q_num_workers=params["q_num_workers"],
        q_verbose=params["q_verbose"],
        sampler=params["patch_sampler"],
        augmentations=params["data_augmentation"],
        preprocessing=params["data_preprocessing"],
        in_memory=params["in_memory"],
        train=False,
    )

    testingDataDefined = True
    if testing_data is None:
        print(
            "No testing data is defined, using validation data for those metrics",
            flush=True,
        )
        testing_data = validation_data
        testingDataDefined = False

    test_data_for_torch = ImagesFromDataFrame(
        testing_data,
        patch_size=params["patch_size"],
        headers=params["headers"],
        q_max_length=params["q_max_length"],
        q_samples_per_volume=params["q_samples_per_volume"],
        q_num_workers=params["q_num_workers"],
        q_verbose=params["q_verbose"],
        sampler=params["patch_sampler"],
        augmentations=params["data_augmentation"],
        preprocessing=params["data_preprocessing"],
        in_memory=params["in_memory"],
        train=False,
    )

    train_dataloader = DataLoader(
        training_data_for_torch,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=params["in_memory"],
    )

    val_dataloader = DataLoader(
        validation_data_for_torch, batch_size=1, pin_memory=params["in_memory"]
    )

    test_dataloader = DataLoader(test_data_for_torch, batch_size=1)

    # Calculat the weights here
    params["weights"] = None

    # Fetch the optimizers
    optimizer = get_optimizer(
        optimizer_name=params["opt"],
        model=model,
        learning_rate=params["learning_rate"],
    )

    scheduler = get_scheduler(
        which_scheduler=params['scheduler'],
        optimizer=optimizer, 
        batch_size=params["batch_size"],
        training_samples_size=len(train_dataloader.dataset), 
        learning_rate=params["learning_rate"]
    )

    # Fetch the appropriate channel keys
    # Getting the channels for training and removing all the non numeric entries from the channels
    batch = next(
        iter(val_dataloader)
    )  # using train_loader makes this slower as train loader contains full augmentations
    all_keys = list(batch.keys())
    channel_keys = []
    value_keys = []
    print("Channel Keys : ", all_keys)
    for item in all_keys:
        if item.isnumeric():
            channel_keys.append(item)
        elif "value" in item:
            value_keys.append(item)
    params["channel_keys"] = channel_keys
    params["value_keys"] = value_keys

    # Start training time here
    start_time = time.time()
    print("\n\n")
    # datetime object containing current date and time
    print("Initializing training at : ", get_date_time())

    # Setup a few loggers for tracking
    train_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "train_logs.csv"),
        metrics=params["metrics"],
    )
    valid_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "valid_logs.csv"),
        metrics=params["metrics"],
    )
    train_logger.write_header(mode="train")
    valid_logger.write_header(mode="valid")

    model, amp, device = send_model_to_device(
        model, amp=params["model"]["amp"], device=params["device"], optimizer=optimizer
    )

    # Setup a few variables for tracking
    best_loss = 1e7
    patience = 0

    # Iterate for number of epochs
    for epoch in range(epochs):

        print("Using device:", device, flush=True)

        # Printing times
        epoch_start_time = time.time()
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        print("Epoch start time : ", get_date_time())

        epoch_train_loss, epoch_train_metric = train_network(
            model, train_dataloader, optimizer, params
        )
        epoch_valid_loss, epoch_valid_metric = validate_network(
            model, val_dataloader, scheduler, params
        )
        patience += 1

        # Write the losses to a logger
        train_logger.write(epoch, epoch_train_loss, epoch_train_metric)
        valid_logger.write(epoch, epoch_valid_loss, epoch_valid_metric)

        print("Epoch end time : ", get_date_time())
        epoch_end_time = time.time()
        print(
            "Time taken for epoch : ",
            (epoch_end_time - epoch_start_time) / 60,
            " mins",
            flush=True,
        )

        # Start to check for loss
        if epoch_valid_loss <= torch.tensor(best_loss):
            best_loss = epoch_valid_loss
            best_train_idx = epoch
            patience = 0
            torch.save(
                {
                    "epoch": best_train_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                os.path.join(output_dir, params['model']['architecture'] + "_best.pth.tar"),
            )

        if patience > params["patience"]:
            print(
                "Performance Metric has not improved for %d epochs, exiting training loop!"
                % (patience),
                flush=True,
            )
            break

    # End train time
    end_time = time.time()

    print(
        "Total time to finish Training : ",
        (end_time - start_time) / 60,
        " mins",
        flush=True,
    )
