import matplotlib
import matplotlib.pyplot as plt
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

'''LINE CURVES'''












file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch0_start0-step1.0-cut10-client5-noniid1.0--budget-1/loss_stats/suro_training_loss.txt"


# Read the values from the text file
with open(file_path, 'r') as file:
    values_full = file.read().split(',')
# Convert the values to floats
values_epoch0 = [float(value) for value in values_full[:-1]]

file_path = "V1-vgg11-cifar10-gan_train_ME_surrogate_earlyepoch50_start0-step1.0-cut10-client5-noniid1.0--budget-1/loss_stats/suro_training_loss.txt"

# Read the values from the text file
with open(file_path, 'r') as file:
    values = file.read().split(',')
# Convert the values to floats
values_epoch50 = []
for i in range(len(values_epoch0)):
    if i > len(values_epoch0) - len(values[:-1]):
        values_epoch50.append(values[i - (len(values_epoch0) - len(values[:-1]))])
    else:
        values_epoch50.append(values[0])
    # values_attacker = [float(value) for value in values[:-1]]






plt.figure(figsize=(8,2))
# plt.hist(values, range = (0, 0.05), bins=20, color="orange")  # You can adjust the number of bins as needed
# plt.plot()  # You can adjust the number of bins as needed
plt.legend()
plt.plot(list(range(len(values_epoch0))), values_epoch50, label='epoch-50')
plt.plot(list(range(len(values_epoch0))), values_epoch0, label='epoch-0')
# Set the labels and title
plt.xlabel('Step')
plt.ylabel('Training Loss')

plt.savefig("surro-loss.pdf", bbox_inches='tight', dpi=600)