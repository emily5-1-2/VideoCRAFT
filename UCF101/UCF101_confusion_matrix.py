from sklearn.metrics import confusion_matrix
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
import numpy as np

def get_classes():
    return ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs',
                'dive', 'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing',
                'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball',
                'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup',
                'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow',
                'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand',
                'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave']

def get_confusion_matrix(model, test_loader):
    model.eval()

    classes = get_classes()

    y_pred = []
    y_true = []

    # iterate over test dataset
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            inputs, labels = data[0], data[-1]
            inputs = inputs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            output = model(inputs)
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save GT

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (48, 30))
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion Matrix for HMDB51')

    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/HMDB51_CM.png')
    return df_cm, cf_matrix