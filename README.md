# Action Modifiers
Code and data for the CVPR 2020 paper ['Action Modifiers: Learning from Adverbs in Instructional Videos'](https://arxiv.org/abs/1912.06617).

## Data

### Training/Test Splits
The files containing the adverb annotations can be found in `train.csv` and `test.csv`. The files contain the following columns:

| Column Name   | Type          | Example | Description |
| ------------- |:-------------:| -------:| -----------:|
| id            | int | 955 | Unique id for this adverb-action annotation |
| vid_id        | string | S7wF6S5ywo4 | YouTube id for the video the annotation is for |
| weak_timestamp | float |  19.435 | Value in seconds of the action-adverb in the narration |
| clustered_adverb | string | quickly | Annotated adverb |
| clustered_action | string | cut | Annotated action |
| task_num | int | 105259 | The id for the task in the HowTo100M dataset |
| adverb | string | fast | The original adverb from the narration |
| action | string | slice | The original action from the narration |

### Features
The features can be downloaded here: https://drive.google.com/open?id=12POBotvtWimAv-PtRswCbUWYucUJ8Aic

This contains two files per entry in `train.csv` or `test.csv`, one for RGB features, one for flow features.

Files are named `<annotation_id>_<modality>.npz`.

### Videos
The videos can be downloaded using: `python utils/download_videos.py <train.csv|test.csv> <download_dir> --trim 20`

The `--trim 20` argument extracts 20 seconds around the weak timestamp as used to extract features.

### Other useful files
`antonym.csv` lists each adverb and its antonym

`adverb_clusters.csv` lists the clusters of adverbs with the following columns: 

| Column Name   | Type          | Example | Description |
| ------------- |:-------------:| -------:| -----------:|
| adverb_id            | int | 0 | ID of this adverb |
| cluster_key        | string | coarsely | Main adverb representing the cluster |
| adverbs | list of strings |  \['coarsley', 'coarse', 'thickly', 'not finely', 'not fine'\] | Narrated adverbs in this cluster |

`action_clusters.csv` is defined similarly


## Code

### Training

To train the model run:
```
python train.py --feature-dir <path_to_directory_containing_features> --checkpoint-dir <path_to_save_checkpoints_to>
```
To train the model without first training the action embedding run
```
python train.py --no-pretrain-action --temporal-agg <sdp|average|single> --feature_dir <path_to_directory_containing_features> --checkpoint-dir <path_to_save_checkpoints_to>
```

### Testing

To test a model run:
```
python test.py --laod <checkpoint_path> --temporal-agg <sdp|average|single> --feature-dir <path_to_features>
```
### Models

Models corresponding to results in the paper can be found under `models/` they are:
* full_model.ckpt - the final result in the paper
* sdp.ckpt - the proposed model without the first stage of only training the action embedding
* average.ckpt - action modifiers without the temporal attention
* single.ckpt - action modifiers with only the second around the weak timestamp
* action.ckpt - a pretrained action embedding with scaled dot-product attention without action modifiers
