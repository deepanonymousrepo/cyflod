from pathlib import Path
import os.path
import pandas as pd
import argparse
import os
import random

def get_args_parser():
    parser = argparse.ArgumentParser(
        'pass', add_help=False)
    parser.add_argument('--datasetpath', default='')
    parser.add_argument('--noisetype', default='sym', choices=['sym', 'asym'], type=str)
    parser.add_argument('--noiseratio', default=0., type=float)
    return parser  # Return the parser object

def generator(args):
    print(args)
    train_dir = args.datasetpath
    training_path = Path(train_dir)
    filepaths = list(training_path.glob(r'**/*.jpg'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    images = pd.concat([filepaths, labels], axis=1)
    train_df = pd.DataFrame(images)
    
    train_df['Label'] = train_df['Label'].map(lambda x: x.lower())
    sorted_df = train_df.sort_values(by='Label').reset_index(drop=True)
    my_list = sorted_df.Label.unique()
    num_cls = len(my_list)
    my_dict = {value: index for index, value in enumerate(my_list)}
    sorted_df['Label_numeric'] = sorted_df['Label'].map(my_dict)
    targets = sorted_df.Label_numeric.tolist()
    per_cls_num = {}
    for class_i in range(num_cls):
        per_cls_num[class_i] = targets.count(class_i)
    noise_label = []
    if args.noisetype == 'asym':
        for label in range(num_cls):
            perClass_num = per_cls_num[label]
            noise_num = int(perClass_num * args.noiseratio)
            for i in range(perClass_num):
                if i < noise_num:
                    if label != num_cls-1:
                        noise_label.append(label + 1)
                    else:
                        noise_label.append(0)
                else:
                    noise_label.append(label)
        
    elif args.noisetype == "sym":
        for label in range(num_cls):
            perClass_num = per_cls_num[label]
            noise_num = int(perClass_num * args.noiseratio)
            for i in range(perClass_num):
                if i < noise_num:
                    noise_label.append(random.randint(0, num_cls-1))
                else:
                    noise_label.append(label)
                    
    sorted_df['Noisy_labels'] = noise_label
    df = sorted_df[['Filepath', 'Noisy_labels', 'Label', 'Label_numeric']]
    dataset_name = args.datasetpath.split('_')[0]
    filename = dataset_name+'_'+args.noisetype+'_'+str(args.noiseratio)+'.csv'
    df.to_csv(filename, index=False)
    print(f"file is create with the name {filename}")
        
    
if __name__ == '__main__':
    parent_parser = get_args_parser()  # Get the parser with arguments
    parser = argparse.ArgumentParser(
        'Noise generator', parents=[parent_parser])  # Incorporate it into the main parser
    args = parser.parse_args()
    generator(args)

    