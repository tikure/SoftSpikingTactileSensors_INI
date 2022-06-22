"""Libraries"""""

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("./Functions")
from Data_functions import *
from Model_functions import *

"""Import Training Data"""""
filenames = ["_AFG_test200"]
model_name = "_AFG_test200_bad"
b15, truths, test_truths, norm_val = import_data(filenames, max_N=100, shape=10,include_norm = False)
# list of filenames, outlier cutoff, fill value for 0 N or "random" (default)
np.savetxt("./Data/norm_val_"+model_name + ".txt", norm_val)
# TODO implement normalized variation

"""Plot to confirm data"""
visualize(b15, test_truths)

"""Setup Model"""
model = vanilla_model(15, feature_dim=40, feat_hidden=[200, 200], activation_fn=nn.ReLU, output_hidden=[200, 200],
                      output_activation=nn.ReLU, feat_activation=None)

# model.load_state_dict(torch.load("./Data/model50k"))
print(model.eval())

"""Setup Data for training"""
# Setup Loader:
batch_size = 250
train_dataset = []
test_dataset = []
for i, inputs in enumerate(b15):
    single_set = [torch.tensor(inputs, dtype=torch.float32), torch.tensor(truths[i], dtype=torch.float32)]
    if not np.random.randint(10) == 9:
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

"""Train MLP"""
num_epochs = 60
# loss & optimizer
learning_rate = 0.001

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
previous_loss = 1
losses = []
overfitting = 0
print("\n")

if True:
    evaluate_MLP(model, test_dataset,title="Before training")
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

        if epoch % 20 == 0:
            print(f'Epoch [{epoch + 1}], Step [{i + 1}/{n_total_steps}], Loss: {epoch_loss}')
            evaluate_MLP(model, test_dataset,title="Training Epoch:"+str(epoch))
            out = [b[2] for b in outputs]
            if sum(out) < 1:
                print("Restarting Model")
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                model = vanilla_model(15, feature_dim=40, feat_hidden=[200, 200], activation_fn=nn.ReLU,
                                      output_hidden=[200, 200],
                                      output_activation=nn.ReLU, feat_activation=None)
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
evaluate_MLP(model, test_dataset)
