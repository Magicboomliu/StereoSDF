import os
import sys
sys.path.append("..")
from utils.utils import read_text_lines
import random
import numpy as np

def select_random_percent(lst, percent=10):
    """
    Selects a specified percentage of items randomly from the list and returns both the selected items and the rest.
    
    Args:
    lst (list): The list from which to select items.
    percent (int): The percentage of the list to randomly select.
    
    Returns:
    tuple: A tuple containing two lists - the randomly selected items and the remaining items.
    """
    # Calculate the number of items to select
    k = int(len(lst) * percent / 100)
    
    # Randomly select 'k' items from the list
    selected_items = random.sample(lst, k)
    
    # Create a set of selected items for fast lookup
    selected_set = set(selected_items)
    
    # Get the remaining items that were not selected
    remaining_items = [item for item in lst if item not in selected_set]
    
    return selected_items, remaining_items

if __name__=="__main__":
    
    list_name = "/home/zliu/NIPS2024/UnsupervisedStereo/StereoSDF/filenames/MPI/MPI_Training.txt"
    lines = read_text_lines(list_name)
    
    classes_dict = {}
    for line in lines:
        if line[:-15] not in classes_dict.keys():
            classes_dict[line[:-15]] = []
    
    
    for line in lines:
        classes_dict[line[:-15]].append(line)
    
    
    select_list = []
    rest_list = []
    for key in classes_dict.keys():
        selected_sub_list, rest_sub_list =select_random_percent(classes_dict[key],percent=10)
        select_list.extend(selected_sub_list)
        rest_list.extend(rest_sub_list)
        
    
    with open("MPI_Val_Sub_List.txt",'w') as f:
        for idx, fname in enumerate(select_list):
            if len(select_list)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)

    with open("MPI_Train_Sub_List.txt",'w') as f:
        for idx, fname in enumerate(rest_list):
            if len(rest_list)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)
    
    print(select_list)
    print(len(select_list))
    print(len(rest_list))        
    
    
