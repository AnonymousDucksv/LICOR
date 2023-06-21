import numpy as np
import pandas as pd
import math
from address import dataset_dir

angle = math.pi/12
t = np.array(range(72)) # 120
theta = angle*t
number_helix = 5 # 10
total_instances = 10000 # 2000

def helix_1_start():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            random_helix_with_flat_idx = 0
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                if helix == random_helix_with_flat_idx:
                    flat_line_start_theta = helix_theta[0]
                    for ang_idx in range(len(helix_theta)):
                        ang = helix_theta[ang_idx]
                        if ang < flat_line_start_theta or ang > flat_line_start_theta + angle*24:
                            continue
                        else:
                            helix_theta[ang_idx] = math.pi/2
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_1_start.csv')

def helix_2_start():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            random_helix_with_flat_idx = np.random.choice([0, 1])
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                if helix == random_helix_with_flat_idx:
                    flat_line_start_theta = helix_theta[0]
                    for ang_idx in range(len(helix_theta)):
                        ang = helix_theta[ang_idx]
                        if ang < flat_line_start_theta or ang > flat_line_start_theta + angle*24:
                            continue
                        else:
                            helix_theta[ang_idx] = math.pi/2
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_2_start.csv')

def helix_1_random_time():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            random_helix_with_flat_idx = 0
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                if helix == random_helix_with_flat_idx:
                    flat_line_start_theta = np.random.choice(helix_theta[:48])
                    for ang_idx in range(len(helix_theta)):
                        ang = helix_theta[ang_idx]
                        if ang < flat_line_start_theta or ang > flat_line_start_theta + angle*24:
                            continue
                        else:
                            helix_theta[ang_idx] = math.pi/2
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_1_random_time.csv')

def helix_2_random_time():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            random_helix_with_flat_idx = np.random.choice([0, 1])
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                if helix == random_helix_with_flat_idx:
                    flat_line_start_theta = np.random.choice(helix_theta[:48])
                    for ang_idx in range(len(helix_theta)):
                        ang = helix_theta[ang_idx]
                        if ang < flat_line_start_theta or ang > flat_line_start_theta + angle*24:
                            continue
                        else:
                            helix_theta[ang_idx] = math.pi/2
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_2_random_time.csv')

def helix_random_feat_random_time():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            random_helix_with_flat_idx = np.random.choice(range(number_helix))
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                if helix == random_helix_with_flat_idx:
                    flat_line_start_theta = np.random.choice(helix_theta[:48])
                    for ang_idx in range(len(helix_theta)):
                        ang = helix_theta[ang_idx]
                        if ang < flat_line_start_theta or ang > flat_line_start_theta + angle*24:
                            continue
                        else:
                            helix_theta[ang_idx] = math.pi/2
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_random_feat_random_time.csv')

def helix_matching_random_amp_feat_time():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            total_number_flat_feat = np.random.choice([2, 3])
            if total_number_flat_feat == 2:
                segment_length = 12
            else: 
                segment_length = 8
            helix_with_flat_idx_list = np.random.choice(range(number_helix), size=total_number_flat_feat, replace=False)
            flat_line_start_idx = np.random.choice(range(48))
            count_number_flat_feat = 0
            for helix in range(number_helix):
                helix_start_theta = np.random.choice(theta)
                helix_theta = helix_start_theta + theta
                if helix in helix_with_flat_idx_list:
                    random_theta_val = np.random.uniform(low=0, high=math.pi)
                    for ang_idx in range(flat_line_start_idx, flat_line_start_idx + segment_length):
                        helix_theta[ang_idx] = random_theta_val
                    flat_line_start_idx += segment_length
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_matching_random_amp_feat_time.csv')

def helix_matching_random_amp_feat_time_no_phase():
    helix_dataset = pd.DataFrame(index=range(total_instances), columns=list(range(len(t)*number_helix))+['label'])
    for instance in range(total_instances):
        all_helix = np.array([])
        if instance%2 == 0:
            for helix in range(number_helix):
                helix_start_theta = theta[0]
                helix_theta = helix_start_theta + theta
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 0
        else:
            total_number_flat_feat = np.random.choice([2, 3])
            if total_number_flat_feat == 2:
                segment_length = 12
            else: 
                segment_length = 8
            helix_with_flat_idx_list = np.random.choice(range(number_helix), size=total_number_flat_feat, replace=False)
            flat_line_start_idx = np.random.choice(range(48))
            count_number_flat_feat = 0
            for helix in range(number_helix):
                helix_start_theta = theta[0]
                helix_theta = helix_start_theta + theta
                if helix in helix_with_flat_idx_list:
                    random_theta_val = np.random.uniform(low=0, high=math.pi)
                    for ang_idx in range(flat_line_start_idx, flat_line_start_idx + segment_length):
                        helix_theta[ang_idx] = random_theta_val
                    flat_line_start_idx += segment_length
                helix_sine = (np.sin(helix_theta) + 1)*0.5
                all_helix = np.concatenate((all_helix, helix_sine))
            target = 1
        helix_dataset.iloc[instance, :-1] = all_helix
        helix_dataset.loc[instance, 'label'] = target
    helix_dataset.to_csv(f'{dataset_dir}helix/helix_matching_random_amp_feat_time_no_phase.csv')






# helix_1_start()
# helix_2_start()
# helix_1_random_time()
helix_2_random_time()
# helix_random_feat_random_time()
#helix_matching_random_amp_feat_time()
#helix_matching_random_amp_feat_time_no_phase()


