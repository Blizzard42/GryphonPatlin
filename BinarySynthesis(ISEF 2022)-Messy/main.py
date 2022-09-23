import tensorflow as tf
import numpy as np
# import prototype_v10_Thesis
# import prototype_v20_Socrates
# import prototype_v21_SocratesThesisTesting
# import prototype_v30_Aristotle
# import prototype_v41_BinarySynthesis
# import prototype_v50_SocratesReborn
# import prototype_v51_AristotleReborn
import prototype_v52_Hegel
# import prototype_v70_BinarySynthesisReborn
# import prototype_v80_LogicalSocrates
# import prototype_v90_Logic1


# ----------------------
# Binary Synthesis Testing
# ----------------------
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# CUTOFF = int(len(x_train)/30)  # Change the integer to adjust the portion of data cut
# WEIGHT = 1
# ADJUSTER = 0
# x_train = x_train[:CUTOFF].copy()
# y_train = y_train[:CUTOFF].copy()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")
#
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000, reshuffle_each_iteration=False).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#
# manual1 = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=[100,10,10,10,10], _epochs=20)
# TRIALS = 1
# for i in range(TRIALS):
#     print("--------------")
#     structure = [1]
#     halt = False
#     while halt is not True:
#         thesis = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=structure, _epochs=1)
#         neo_structure = structure.copy()
#         neo_structure[len(neo_structure)-1] += 1
#         thesis2 = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=neo_structure, _epochs=1)
#         if thesis2.test_trainset_accuracy.result() - thesis.test_trainset_accuracy.result() > -0.02:
#             structure = neo_structure
#             print('Expanding Horizontally')
#             print(f'New Structure: {structure}')
#         else:
#             neo_structure = structure.copy()
#             neo_structure.append(structure[len(structure)-1])
#             thesis3 = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=neo_structure, _epochs=1)
#             if thesis3.test_trainset_accuracy.result() - thesis.test_trainset_accuracy.result() > -0.02:
#                 structure = neo_structure
#                 print('Expanding Vertically')
#                 print(f'New Structure: {structure}')
#             else:
#                 halt = True

# ----------------------
# Hegel Testing
# ----------------------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
CUTOFF = int(len(x_train)/15)  # Change the integer to adjust the portion of data cut
WEIGHT = 1
ADJUSTER = 0
x_train = x_train[:CUTOFF].copy()
y_train = y_train[:CUTOFF].copy()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000, reshuffle_each_iteration=False).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

TRIALS = 100
combined_differences=[]
combined_differences2=[]
combined_differences3=[]
combined_synthesis_accuracies=[]
combined_direct_thesis_accuracies=[]
for j in [96]: # 1,32,64,96,128
    accuracy_difference = []
    accuracy_difference2 = []
    accuracy_difference3 = []
    synthesis_accuracies_per_epoch = []
    direct_accuracies_per_epoch = []
    for i in range(TRIALS):
        print("--------------")
        thesis = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=[j], _epochs=3)
        # thesis2 = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=[j], _epochs=3)
        antithesis = prototype_v52_Hegel.antithesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=[j], _epochs=3, _sample_weights=thesis.output_weights)
        print('Synthesis Test:')
        synthesis = prototype_v52_Hegel.synthesize(_train_ds=train_ds, _test_ds=test_ds, _epochs=50,
                                                   _thesis_model=thesis.model, _antithesis_model=antithesis.model)
        # synthesis2 = prototype_v52_Hegel.synthesize(_train_ds=train_ds, _test_ds=test_ds, _epochs=3,
        #                                            _thesis_model=thesis.model, _antithesis_model=thesis2.model)
        # double_thesis = prototype_v52_Hegel.synthesize(_train_ds=train_ds, _test_ds=test_ds, _epochs=3,
        #                                            _thesis_model=thesis.model, _antithesis_model=thesis.model)
        thesis_direct = prototype_v52_Hegel.thesize(_train_ds=train_ds, _test_ds=test_ds, _network_size=[2*j], _epochs=50)
        # accuracy_difference.append((synthesis.test_accuracy.result() - synthesis2.test_accuracy.result()).numpy())
        # accuracy_difference2.append((synthesis.test_accuracy.result() - double_thesis.test_accuracy.result()).numpy())
        synthesis_accuracies_per_epoch.append(synthesis.combined_test_accuracies)
        direct_accuracies_per_epoch.append(thesis_direct.combined_test_accuracies)
        accuracy_difference3.append((synthesis.test_accuracy.result() - thesis_direct.test_accuracy.result()).numpy())
    synthesis_accuracies_per_epoch = np.array(synthesis_accuracies_per_epoch)
    synthesis_std_per_epoch = np.std(synthesis_accuracies_per_epoch.T, axis=1)
    print(f'Synthesis Standard Deviation Per Epoch: {synthesis_std_per_epoch}')
    synthesis_accuracies_per_epoch = np.mean(synthesis_accuracies_per_epoch.T, axis=1)
    print(f'Synthesis Means Per Epoch: {synthesis_accuracies_per_epoch}')
    direct_accuracies_per_epoch = np.array(direct_accuracies_per_epoch)
    direct_std_per_epoch = np.std(direct_accuracies_per_epoch.T, axis=1)
    print(f'Direct Thesis Standard Deviation Per Epoch: {direct_std_per_epoch}')
    direct_accuracies_per_epoch = np.mean(direct_accuracies_per_epoch.T, axis=1)
    print(f'Direct Thesis Means Per Epoch: {direct_accuracies_per_epoch}')


    # combined_differences.append(accuracy_difference)
    # combined_differences2.append(accuracy_difference2)
    combined_differences3.append(accuracy_difference3)
# print(f'Combined Differences (Indirect Synthesis): {combined_differences}')
# print(f'Combined Differences (Double Thesis): {combined_differences2}')
print(f'Combined Differences (Direct Thesis): {combined_differences3}')
# mean = np.mean(combined_differences, axis=1)
# mean2 = np.mean(combined_differences2, axis=1)
mean3 = np.mean(combined_differences3, axis=1)
# std = np.std(combined_differences, axis=1)
# std2 = np.std(combined_differences2, axis=1)
std3 = np.std(combined_differences3, axis=1)
# print(f'Mean (Indirect Synthesis): {mean}')
# print(f'Mean (Double Thesis): {mean2}')
print(f'Mean (Direct Thesis): {mean3}')
# print(f'Standard Deviation (Indirect Synthesis): {std}')
# print(f'Standard Deviation (Double Thesis): {std2}')
print(f'Standard Deviation (Direct Thesis): {std3}')




# ----------------------
# Socrates Reborn Thesis Testing
# ----------------------
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# CUTOFF = int(len(x_train)/30)  # Change the integer to adjust the portion of data cut
# WEIGHT = 1
# ADJUSTER = 0
# x_train = x_train[:CUTOFF].copy()
# y_train = y_train[:CUTOFF].copy()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")
#
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000, reshuffle_each_iteration=False).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#
#
# TRIALS = 10
# total_thesis = 0
# total_antithesis = 0
# total_shared = 0
# for i in range(TRIALS):
#     print('----------------')
#     print(f'Trial {i+1}: ')
#     control = prototype_v50_SocratesReborn.trial(train_ds, test_ds, WEIGHT, ADJUSTER, network_size=1)
#     thesis, antithesis, shared = prototype_v50_SocratesReborn.trial(train_ds, test_ds, WEIGHT, ADJUSTER, network_size=1)
#     total_thesis += thesis
#     total_antithesis += antithesis
#     total_shared += shared
# total_thesis /= TRIALS
# total_antithesis /= TRIALS
# total_shared /= TRIALS
# print(f'Total Thesis: {total_thesis}')
# print(f'Total Antithesis: {total_antithesis}')
# print(f'Total Shared: {total_shared}')


# ----------------------
# Aristotle Reborn Testing
# ----------------------
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# CUTOFF = int(len(x_train)/30)  # Change the integer to adjust the portion of data cut
# WEIGHT = 1
# ADJUSTER = 0
# x_train = x_train[:CUTOFF].copy()
# y_train = y_train[:CUTOFF].copy()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")
#
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000, reshuffle_each_iteration=False).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#
# TRIALS = 5
# total = 0
# total2 = 0
# for i in range(TRIALS):
#     print("--------------")
#     thesis_model, antithesis_model, thesis_score, antithesis_score, thesis_result = prototype_v50_SocratesReborn.trial(train_ds, test_ds, WEIGHT, ADJUSTER)
#     aristotle = prototype_v51_AristotleReborn.trial(train_ds, test_ds, WEIGHT, ADJUSTER, thesis_model, antithesis_model,
#                                                     thesis_score, antithesis_score)
#     total += aristotle
#     total2 += thesis_result
# total /= TRIALS
# total2 /= TRIALS
# print(f'Thesis Total: {total2}')
# print(f'Aristotle Total: {total}')


# # ----------------------
# # Logic1 Testing
# # ----------------------
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# CUTOFF = int(len(x_train)/30)  # Change the integer to adjust the portion of data cut
# WEIGHT = 1
# ADJUSTER = 0
# x_train = x_train[:CUTOFF].copy()
# y_train = y_train[:CUTOFF].copy()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")
#
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000, reshuffle_each_iteration=False).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#
# TRIALS = 1
# for i in range(TRIALS):
#     thesis_model, antithesis_model, thesis_score, antithesis_score, thesis_result = prototype_v90_Logic1.trial(train_ds, test_ds, WEIGHT, ADJUSTER)


# ----------------------
# Extensive Thesis Testing
# ----------------------
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# CUTOFF = [1000, 2000, 4000, 8000, 16000, 32000, 60000]
# train_sets = []
# for i in CUTOFF:
#     x_set = x_train[:i].copy()
#     y_set = y_train[:i].copy()
#     x_set = x_set / 255.0
#     x_set = x_set[..., tf.newaxis].astype("float32")
#     train_sets.append(tf.data.Dataset.from_tensor_slices(
#         (x_set, y_set)).shuffle(10000, reshuffle_each_iteration=False).batch(32))
#
# x_test = x_test / 255.0
# x_test = x_test[..., tf.newaxis].astype("float32")
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#
#
# trials = 50
# epochs = 10
# total_data = []
# for i in train_sets:
#     data = []
#     for j in [1, 2, 4, 8, 16, 32, 64, 96, 128]:
#         results = []
#         for k in range(trials):
#             results.append(prototype_v10_Thesis.trial(i, test_ds, network_size=j, epochs=epochs))
#         results = np.array(results).T.tolist()
#         for record_loop_var in range(epochs):
#             data.append({"Stamp": f"Sample Size: {len(i)/0.03125}", "Network Size": j, "Epochs": record_loop_var + 1,
#                         "Max": np.max(results[record_loop_var]),
#                          "Mean": np.mean(results[record_loop_var]),
#                          "Standard Deviation": np.std(results[record_loop_var]),
#                          "Max Z-Score": (np.max(results[record_loop_var]) - np.mean(results[record_loop_var])) /
#                                         np.std(results[record_loop_var]),
#                          "Results": results[record_loop_var]})
#     total_data.append(data)
#
# for i in total_data:
#     print("----------------------")
#     for j in i:
#         print(j)


# ----------------------
# v10_Thesis Test
# ----------------------
# TRIALS = 5
# total = 0
# for i in range(TRIALS):
#     accuracy = prototype_v10_Thesis.trial(epochs=3)
#     total += accuracy
# total /= TRIALS
# print(f'Total: {total}')

# Socrates v21 and Versions past v41
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# CUTOFF = int(len(x_train)/30)  # Change the integer to adjust the portion of data cut
# WEIGHT = 1
# ADJUSTER = 0
# x_train = x_train[:CUTOFF].copy()
# y_train = y_train[:CUTOFF].copy()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")
#
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000, reshuffle_each_iteration=False).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# ----------------------
# Binary Synthesis Testing
# ----------------------
# TRIALS = 10
# former_errors = [True]
# for i in range(TRIALS):
#     former_errors = prototype_v70_BinarySynthesisReborn.trial(train_ds, test_ds, WEIGHT, ADJUSTER, former_errors, int(i % 2))

# import matplotlib.pyplot as plt
# b1, = train_ds.take(1)
# im1 = plt.imshow(b1[0][0])
# plt.show()
# b1, = train_ds.take(1)
# im2 = plt.imshow(b1[0][0])
# plt.show()
# for images, labels in train_ds:
#     img = plt.imshow(images[0])
#     break
# plt.show()
# for images, labels in train_ds:
#     img = plt.imshow(images[0])
#     break


# ----------------------
# Binary Synthesis Testing
# ----------------------
# TRIALS = 1
# SYNLAYERS = 1
# for i in range(TRIALS):
#     print('Binary Synthesis: ')
#     former_predictions = np.zeros([1, 60000, 10])
#     blind_spots = np.array([True])
#     for j in range(SYNLAYERS):
#         former_predictions, blind_spots = prototype_v41_BinarySynthesis.trial(former_predictions, blind_spots, epochs=1)
#
#     print('Thesis:')
#     prototype_v10_Thesis.trial(epochs=6)
#     print('')

# TRIALS = 3
# for i in range(TRIALS):
#     thesis, antithesis, sample_weights, anti_sample_weights = prototype_v20_Socrates.trial()
#     thesis.trainable = False
#     antithesis.trainable = False
#     aristotle_accuracy = prototype_v30_Aristotle.trial(thesis, antithesis, sample_weights, anti_sample_weights)
#
#     # thesis_accuracy = prototype_v10_Thesis.trial(1)
#     #
#     # print(f'Thesis: {thesis_accuracy}')
#     print(f'Aristotle: {aristotle_accuracy}')
#     print('')
