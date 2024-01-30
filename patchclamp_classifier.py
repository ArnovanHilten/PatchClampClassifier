import pyabf
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump
from joblib import load


from feature_extraction import correct_abf_signal, low_pass_filter, smooth_abf_signal
from feature_extraction import check_single_pulse, extract_features


def main(datapath = "/Users/drv850/Google Drive/My Drive/Postdoc/data_juliet/",
         metadata_path = "/Users/drv850/Google Drive/My Drive/Postdoc/data_juliet/metadata.csv",
         cutoff = 100,
         fs = 10000, start_pulse_offset = 400, end_point = 6000 , feature_save_path = None):


    metadata = pd.read_csv(metadata_path)
    print(metadata.head())


    pd_features, pd_features_total = get_all_feature(datapath, metadata, start_pulse_offset, end_point, cutoff, fs)
    
    if feature_save_path is not(None):
        pd_features_total.to_csv(feature_save_path + "metadata_features.csv", index=False)

    random_forst_classification(pd_features, pd_features_total, train=True)


def apply_random_forest(
         datapath = "/Users/drv850/Google Drive/My Drive/Postdoc/data_juliet/",
         filename = "2022_02_01_0015",
         cutoff = 100, fs = 10000, start_pulse_offset = 400, end_point = 6000 , feature_save_path = None):


    features_signal, feat_columns = get_feature(datapath, filename, start_pulse_offset, end_point, cutoff, fs)
    
    feature_columns = list(feat_columns) + ["std_"+ x for x in list(feat_columns) ]
    total_feature_columns = ["raw_"+ x for x in list(feature_columns) ] + ["smooth_"+ x for x in list(feature_columns)  ]+ ["low_"+ x for x in list(feature_columns) ]

    pd_features = pd.DataFrame(np.reshape(features_signal, (1,-1)), columns=total_feature_columns)
    clf = RandomForestClassifier(n_estimators=2000, random_state=0)
    clf = load('random_forest_model.joblib')

    y_probs = clf.predict_proba(pd_features)[:, 1]  
    print(y_probs[0] , filename)

    if feature_save_path is not(None):
        pd_features.to_csv(feature_save_path +'/'+str(filename) + "_features.csv", index=False)

    return y_probs[0] 


def get_all_feature(datapath, metadata, start_pulse_offset, end_point, cutoff, fs):
    total_features = []
    for filename in tqdm.tqdm(metadata["filename"]):
        features_signal, feat_columns = get_feature(datapath, filename, start_pulse_offset, end_point, cutoff, fs)
        total_features.append(features_signal)
    
    pd_features = pd.DataFrame(total_features)
    
    
    feature_columns = list(feat_columns) + ["std_"+ x for x in list(feat_columns) ]
    total_feature_columns = ["raw_"+ x for x in list(feature_columns) ] + ["smooth_"+ x for x in list(feature_columns)  ]+ ["low_"+ x for x in list(feature_columns) ]
    pd_features.columns = total_feature_columns 

    pd_features_total = pd.concat([metadata, pd_features], axis=1)   

    return pd_features, pd_features_total



def get_feature(datapath, filename, start_pulse_offset, end_point, cutoff, fs):
    abf = pyabf.ABF(datapath + "/"+ str(filename)+".abf")

    start_pulse_signal = np.argmax(abs(abf.sweepC))
    start_pulse_with_offset = start_pulse_signal+ start_pulse_offset

    if not(check_single_pulse(abf)):
        print("error")

    pulse_signal = np.argmax(abs(abf.sweepC)) * abf.dataSecPerPoint
    time_to_peak = np.argmax(abs(abf.sweepY)) * abf.dataSecPerPoint

    corrected_signal = correct_abf_signal(abf)
    lowpass_signal = low_pass_filter(corrected_signal,cutoff, fs)
    smoothend_signal = smooth_abf_signal(corrected_signal)

    s = lowpass_signal[:,start_pulse_signal:end_point]
    pd_features_signal_low = extract_features(s, start_pulse_with_offset)
    features_signal_low = np.concatenate([pd_features_signal_low.mean(0).values, pd_features_signal_low.std(0).values])
    s = corrected_signal[:,start_pulse_signal:end_point]
    pd_features_signal_raw = extract_features(s, start_pulse_with_offset)
    features_signal_raw = np.concatenate([pd_features_signal_raw.mean(0).values, pd_features_signal_raw.std(0).values])
    s = smoothend_signal[:,start_pulse_signal:end_point]
    pd_features_signal = extract_features(s, start_pulse_with_offset)
    features_signal_smooth = np.concatenate([pd_features_signal.mean(0).values, pd_features_signal.std(0).values])

    features_signal = np.concatenate([features_signal_raw, features_signal_smooth , features_signal_low])
    return features_signal, pd_features_signal_raw.columns



def random_forst_classification(pd_features, pd_features_total, train=True):

    # Splitting the pd_features into features and outcome
    X = pd_features  # Features (excluding the 'Input' column)
    y = pd_features_total['Class'] # Outcome (the 'Input' column)

    # Splitting the dataset into training (80%) and a temporary set (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

    # Splitting the temporary set into validation and test sets (each 10% of the total dataset)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Checking the distribution in each set
    distribution_train = y_train.value_counts(normalize=True)
    distribution_val = y_val.value_counts(normalize=True)
    distribution_test = y_test.value_counts(normalize=True)

    (distribution_train, distribution_val, distribution_test)
    

    clf = RandomForestClassifier(n_estimators=2000, random_state=0)

    if train==False:
        loaded_model = load('random_forest_model.joblib')
    else:
        clf.fit(X_train, y_train)
        dump(clf, 'random_forest_model.joblib')


    # Predict probabilities for the validation set
    y_probs = clf.predict_proba(X_val)[:, 1]  # get the probabilities for the positive class

    # Calculate ROC curve and ROC AUC
    fpr, tpr, thresholds = roc_curve(y_val, y_probs)
    roc_auc = roc_auc_score(y_val, y_probs)

    print("AUC validaiton", roc_auc)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROCAUC_random_forest.png', dpi=300, bbox_inches='tight')
    

    # Assuming clf is your trained RandomForestClassifier model
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print(f"{f + 1}. feature {pd_features.columns[indices[f]]} ({importances[indices[f]]})")

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])


    
if __name__ == "__main__":

    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Process and apply random forest to dataset.")
    subparsers = parser.add_subparsers(dest="command")

    # Create a subparser for the main function
    parser_main = subparsers.add_parser('main', help='Run the main processing and classification function.')
    parser_main.add_argument('--datapath', type=str, default="/Users/drv850/Google Drive/My Drive/Postdoc/data_juliet", help='Path to the data directory')
    parser_main.add_argument('--metadata_path', type=str, default="/Users/drv850/Google Drive/My Drive/Postdoc/data_juliet/metadata.csv", help='Path to the metadata CSV file')
    parser_main.add_argument('--cutoff', type=int, default=100, help='Cutoff parameter')
    parser_main.add_argument('--fs', type=int, default=10000, help='Sampling frequency')
    parser_main.add_argument('--start_pulse_offset', type=int, default=400, help='Start pulse offset')
    parser_main.add_argument('--end_point', type=int, default=6000, help='End point for processing')
    parser_main.add_argument('--feature_save_path', type=str, default=None, help='Path to save extracted features')

    # Create a subparser for the apply_random_forest function
    parser_apply_rf = subparsers.add_parser('apply_rf', help='Apply the random forest model to a new dataset.')
    parser_apply_rf.add_argument('--datapath', type=str, default="/Users/drv850/Google Drive/My Drive/Postdoc/data_juliet", help='Path to the data directory')
    parser_apply_rf.add_argument('--filename', type=str, default="2022_02_01_0015.abf", help='Filename to process')
    parser_apply_rf.add_argument('--cutoff', type=int, default=100, help='Cutoff parameter')
    parser_apply_rf.add_argument('--fs', type=int, default=10000, help='Sampling frequency')
    parser_apply_rf.add_argument('--start_pulse_offset', type=int, default=400, help='Start pulse offset')
    parser_apply_rf.add_argument('--end_point', type=int, default=6000, help='End point for processing')
    parser_apply_rf.add_argument('--feature_save_path', type=str, default=None, help='Path to save extracted features')
    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the subcommand
    if args.command == 'main':
        main(datapath = args.datapath,
                            metadata_path = args.metadata_path,
                            cutoff = args.cutoff,
                            fs = args.fs,
                            start_pulse_offset = args.start_pulse_offset,
                            end_point = args.end_point,
                            feature_save_path = args.feature_save_path)
    elif args.command == 'apply_rf':
        apply_random_forest(datapath = args.datapath,
                            filename = args.filename,
                            cutoff = args.cutoff,
                            fs = args.fs,
                            start_pulse_offset = args.start_pulse_offset,
                            end_point = args.end_point,
                            feature_save_path = args.feature_save_path)