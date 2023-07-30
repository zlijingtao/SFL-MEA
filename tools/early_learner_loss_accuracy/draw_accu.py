import matplotlib
import matplotlib.pyplot as plt
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

'''LINE CURVES'''

def extract_surro_val_accu_from_log(file_path):
    accuracy_list = []
    pass
    #we read lines from log< search for "Test (surrogate):	Loss " and mark a flag to True, and in the next line, we extract the val accurayc
    search_text = "Test (surrogate):	Loss "
    with open(file_path, 'r') as file:
        found_surro_flag = False
        line_number = 0
        for line in file:

            if found_surro_flag:
                # val accuracy is in the current line.
                accuracy = float(line.strip().split("* Prec@1 ")[-1])
                accuracy_list.append(accuracy)
                found_surro_flag = False
            if search_text in line:
                # print(f"Found '{search_text}' in line {line_number}: {line.strip()}")
                found_surro_flag = True
            line_number += 1
    
    return accuracy_list





file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch0_test_consistency_start0-step1.0-cut10-client5-noniid1.0--budget-1/train.log"


# Read the values from the text file
accuracy_epoch0_consist = extract_surro_val_accu_from_log(file_path)




file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch0_start0-step1.0-cut10-client5-noniid1.0--budget-1/train.log"


# Read the values from the text file
accuracy_epoch0 = extract_surro_val_accu_from_log(file_path)

file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch50_start0-step1.0-cut10-client5-noniid1.0--budget-1/train.log"
accuracy_epoch50 = extract_surro_val_accu_from_log(file_path)
# accuracy_epoch50 = [10.0 for _ in range(150)].extend(accuracy_epoch50)

file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch100_start0-step1.0-cut10-client5-noniid1.0--budget-1/train.log"

accuracy_epoch100 = extract_surro_val_accu_from_log(file_path)


file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch150_start0-step1.0-cut10-client5-noniid1.0--budget-1/train.log"

accuracy_epoch150 = extract_surro_val_accu_from_log(file_path)


plt.figure(figsize=(8,2))
# plt.hist(values, range = (0, 0.05), bins=20, color="orange")  # You can adjust the number of bins as needed
# plt.plot()  # You can adjust the number of bins as needed
plt.plot(list(range(len(accuracy_epoch0))), accuracy_epoch0_consist, label='fixed-target')
plt.plot(list(range(len(accuracy_epoch0))), accuracy_epoch0, label='epoch-0')
plt.plot(list(range(len(accuracy_epoch0))), accuracy_epoch50, label='epoch-50')
plt.plot(list(range(len(accuracy_epoch0))), accuracy_epoch100, label='epoch-100')
plt.plot(list(range(len(accuracy_epoch0))), accuracy_epoch150, label='epoch-150')
# Set the labels and title
plt.xlabel('Step')
plt.ylabel('Validation Accuracy')
# plt.legend()
plt.savefig("surro-accu.svg", bbox_inches='tight', dpi=600)

# plt.figure(figsize=(8,2))
# # plt.hist(values, range = (0, 0.05), bins=20, color="orange")  # You can adjust the number of bins as needed
# # plt.plot()  # You can adjust the number of bins as needed
# plt.legend()
# plt.plot(list(range(len(values_epoch0))), values_epoch50, label='epoch-50')
# plt.plot(list(range(len(values_epoch0))), values_epoch0, label='epoch-0')
# # Set the labels and title
# plt.xlabel('Step')
# plt.ylabel('Training Loss')

# plt.savefig("surro-loss.png", bbox_inches='tight', dpi=600)