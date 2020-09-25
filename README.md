# Sound-Signal-SED
Automatic recognition and detection of sound signals

The framework can realize the recognition and detection of acoustic signals. It's modified by DCASE2019 task3, the DOA is removed. More description 
of this task can be found in http://dcase.community/challenge2019/task-sound-event-localization-and-detection.


Requirements
python 3.7 + pytorch 1.6

Run the project with:
(1) Modify the paths of dataset and your workspace
(2) Extract features: features.py with 'calculate_feature_for_each_audio_file' and then 'calculate_scalar'
(3) Train model: train.py with 'train' 
(4) Inference: train.py with 'inference_validation'

## Function
We apply convolutional neural networks using the log mel spectrogram with singal channels and multi channels. It can be used for array signal processing subsequently.
The label targets are categories and onset and offset times.
The model with 5/9/13 layers CNN. To train a CNN with 9 layers and a mini-batch size of 32, the training takes approximately 200 ms / iteration on a single card GTX Titan Xp GPU.
The model is trained for 5000 iterations.
