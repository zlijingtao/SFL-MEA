import matplotlib
import matplotlib.pyplot as plt
import numpy as np
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)



'''Histogram'''



# Specify the path to your text file
# file_path = "training_loss_client4_new.txt"

# # Read the values from the text file
# with open(file_path, 'r') as file:
#     values = file.read().split(',')
# # Convert the values to floats
# values = [float(value) for value in values[:-1]]

# # Generate the histogram
# # plt.hist(values, range = (0, 0.05), bins=20, color="orange")  # You can adjust the number of bins as needed
# plt.hist(values, range = (0, 0.05), bins=20)  # You can adjust the number of bins as needed

# # Set the labels and title
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# # plt.title('Histogram of Values')

# # Display the histogram
# plt.savefig("client4-loss-histogram.png", bbox_inches='tight', dpi=600)



'''LINE CURVES'''

file_path = "training_loss_client0_new.txt"


# Read the values from the text file
with open(file_path, 'r') as file:
    values = file.read().split(',')
# Convert the values to floats
values_benign = [float(value) for value in values[:-1]]

file_path = "training_loss_client4_new.txt"

# Read the values from the text file
with open(file_path, 'r') as file:
    values = file.read().split(',')
# Convert the values to floats
values_attacker = [float(value) for value in values[:-1]]
plt.figure(figsize=(12,5))
# Generate the histogram
# plt.hist(values, range = (0, 0.05), bins=20, color="orange")  # You can adjust the number of bins as needed
# plt.plot()  # You can adjust the number of bins as needed
plt.plot(list(range(len(values_attacker))), values_attacker, label='Attacker')
plt.plot(list(range(len(values_attacker))), values_benign, label='Benign Clints')
# Set the labels and title
plt.xlabel('Step')
plt.ylabel('Training Loss')
# plt.title('Histogram of Values')

# Display the histogram
plt.savefig("client4-loss-histogram.png", bbox_inches='tight', dpi=600)