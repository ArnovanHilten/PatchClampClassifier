# PatchClamp data with Random forest classifier
## Install:
```
git@github.com:ArnovanHilten/PatchClampClassifier.git
```

```
conda env create -f environment_patchclamp.yml
```

## Retraining the random forest
```
python patchclamp_classifier.py main
```
Look at help with

```
python patchclamp_classifier.py main --help
```
output
```

usage: patchclamp_classifier.py main [-h] [--datapath DATAPATH] [--metadata_path METADATA_PATH] [--cutoff CUTOFF] [--fs FS] [--start_pulse_offset START_PULSE_OFFSET]
                                     [--end_point END_POINT] [--feature_save_path FEATURE_SAVE_PATH]

options:
  -h, --help            show this help message and exit
  --datapath DATAPATH   Path to the data directory
  --metadata_path METADATA_PATH
                        Path to the metadata CSV file
  --cutoff CUTOFF       Cutoff parameter
  --fs FS               Sampling frequency
  --start_pulse_offset START_PULSE_OFFSET
                        Start pulse offset
  --end_point END_POINT
                        End point for processing
  --feature_save_path FEATURE_SAVE_PATH
                        Path to save extracted feature
```


## Extracting the features for a new ABF and applying the random forest:
```
python patchclamp_classifier.py apply_rf  --filename "2022_02_01_0015"  --feature_save_path "./"
```

```

python patchclamp_classifier.py apply_rf --help                                                         PatchClamp 14:34:31

usage: patchclamp_classifier.py apply_rf [-h] [--datapath DATAPATH] [--filename FILENAME] [--cutoff CUTOFF] [--fs FS] [--start_pulse_offset START_PULSE_OFFSET]
                                         [--end_point END_POINT] [--feature_save_path FEATURE_SAVE_PATH]

options:
  -h, --help            show this help message and exit
  --datapath DATAPATH   Path to the data directory
  --filename FILENAME   Filename to process
  --cutoff CUTOFF       Cutoff parameter
  --fs FS               Sampling frequency
  --start_pulse_offset START_PULSE_OFFSET
                        Start pulse offset
  --end_point END_POINT
                        End point for processing
  --feature_save_path FEATURE_SAVE_PATH
                        Path to save extracted features

```
