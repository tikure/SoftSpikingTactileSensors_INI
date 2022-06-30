"""Libraries"""""

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("./Functions")
from Data_functions import *
from Model_functions import *

"""Import Training Data"""""
filenames = ["_AFG_Board1_50","_AFG_Board2_50"]
#filenames = ["_AFG_board2_50_screw","_AFG_board2_50_2_screw"]
model_name = "_AFG_Board_double_2"
b15, truths, test_truths, norm_val, b15_norm = import_data(filenames = ["_AFG_board1_50"], max_N=100, shape="random", include_norm = False,
                                                 normalization ='divisive', data_count_percent = 100)

b152, truths2, test_truths2, norm_val2, b15_norm2 = import_data(filenames = ["_AFG_board2_50"], max_N=100, shape="random", include_norm = False,
                                                 normalization ='divisive', data_count_percent = 100)

np.savetxt("./Data/norm_val_"+model_name + ".txt", norm_val)

b15 = b15 + b152
truths = truths +truths2
test_truths = test_truths + test_truths2
norm_val = norm_val
b15_norm = b15_norm + b15_norm2


# list of filenames, outlier cutoff, fill value for 0 N or "random" (default)

# TODO implement normalized variation
# DONE implement local normalization - amplifies only very small parts, not training efficient
# DONE implement subtractive normalization - sensors with higher norm val are also have higher variance so divisive is preffereable
# TODO comments on Sensor Viewer
# DONE temporal selection of testing data - changes little about training accuracy
# DONE sigmoid output - implmenented but does not train well
# DONE new sensors - board2 is same as original board


"""Plot to confirm data"""
visualize(b15, test_truths, b15_norm)

"""Setup Model"""
def setup_model():
    model = vanilla_model(15, feature_dim=40, feat_hidden=[200, 200], activation_fn=nn.ReLU, output_hidden=[200, 200],
                  output_activation=nn.ReLU, feat_activation=None, scaled_sigmoid=scaled_sigmoid)
    return model
    # model.load_state_dict(torch.load("./Data/model50k"))

scaled_sigmoid = False #adds scaled sigmoid to output (0-20) #CURRENTLY BREAKS MODEL
sorting = "temporal" #Is testing data removed randomly or from the end, can be random or temporal
batch_size = 250
model = setup_model()
print(model.eval())

"""Setup Data for training"""
# Setup Loader:
train_dataset = []
test_dataset = []

if sorting == "temporal":
    for i, inputs in enumerate(b15):
        single_set = [torch.tensor(inputs, dtype=torch.float32), torch.tensor(truths[i], dtype=torch.float32)]
        if not np.random.randint(10) == 9:
            train_dataset.append(single_set)
        else:
            test_dataset.append(single_set)
elif sorting == "temporal":
    for i, inputs in enumerate(b15):
        single_set = [torch.tensor(inputs, dtype=torch.float32), torch.tensor(truths[i], dtype=torch.float32)]
        if i < len(b15)* (9/10):# last 10% go to testing data
            train_dataset.append(single_set)
        else:
            test_dataset.append(single_set)

print("Training Set: ", len(train_dataset))
print("Testing Set: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

examples = iter(train_loader)
example_data, example_targets = examples.next()
print(example_data[0])
print(example_targets[0])

"""Setup Training"""
num_epochs = 120
learning_rate = 0.001
criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
prefitted_MLP = False

"""Train MLP"""
n_total_steps = len(train_loader)
previous_loss = 1
losses = []
overfitting = 0
print("\n")

if not prefitted_MLP:
    #evaluate_MLP(model, test_dataset,title="Before training")
    print("Beginning Training")
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (b, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(b)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        epoch_loss = round(train_loss / len(train_dataset), 5)

        if epoch < 120:
            sum = 0
            for o in outputs:
                sum += o[2]
            #print(sum)
            if sum.item() < 5:
                #print("Restarting Model")
                model = setup_model()
                epoch = -1
                previous_loss = -1
                losses = []
                overfitting = 0
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if epoch % 20 == 0:
            print(f'Epoch [{epoch + 1}], Step [{i + 1}/{n_total_steps}], Loss: {epoch_loss}')
            evaluate_MLP(model, test_dataset,title="Training Epoch:"+str(epoch))
            out = [b[2] for b in outputs]
        delta_loss = previous_loss - epoch_loss
        if delta_loss < 0.01 * epoch_loss + 0.0001:
            overfitting += 1
            # print(f"Overfitting begun ({overfitting})")
        else:
            overfitting -= 1
        losses.append(epoch_loss)
        previous_loss = epoch_loss
        if overfitting > 3:
            print(f"Done, final Loss: {epoch_loss}, in {epoch} Epochs")
            break
    plt.plot(losses)
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Epoch Loss")
    plt.show()
    torch.save(model.state_dict(), "./Data/MLP_"+model_name)
else:
    model.load_state_dict(torch.load("./Data/MLP_"+model_name))


"""Plot error for comprehension"""
print(model.eval())
title = "Final "+ model_name + " \n Sorting: "+ sorting + " \n Scaling: " + str(scaled_sigmoid)
evaluate_MLP(model, test_dataset, title = title )
