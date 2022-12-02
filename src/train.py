import random
import re
import torch
import torch.optim as optim
from network import DocumentNetFc
import loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def collate_fn_batch(batch, pct, vector_size):
    new_batch = []
    vectors = []
    labels = []

    for pair in batch:
        for vec1, vec2 in pair:
            # Randomly select which pairs of sets to transfer elements between and which pairs not to touch
            transfer_pair = torch.randint(0, 2, (1,)) > 0

            if transfer_pair:
                # Decide which vector to take elements from
                set_selection = torch.randint(0, 1, (1,))
                set_take = set(vec1)
                set_give = set(vec2)

                if set_selection == 1:
                    temp = set_take
                    set_take = set_give
                    set_give = temp

                # Calculate num elements to select
                k = int(pct * len(set_take))

                # Randomly select x% of elements from vec1/vec2 to take
                rand_elements = random.sample(set_take, k)

                # Randomly sample the same number of elements in the other set to remove
                # orig_set_give = set_give.deepcopy()
                orig_set_give = set_give.copy()

                remove = random.sample(set_give, min(k, len(set_give)))
                set_give.difference_update(remove)
                set_give.union(rand_elements)

                # Set vec1, vec2 equal to the newly modified sets
                vec1 = list(orig_set_give)
                vec2 = list(set_give)
            
            # Compute the new label for each pair
            set_vec1 = set(vec1)
            set_vec2 = set(vec2)

            labels.append(0 if len(set_vec1.intersection(set_vec2)) / len(set_vec1.union(set_vec2)) < 0.8 else 1)

            # Convert these sets to vectors with 0's and 1's
            vec1_filter = [0 for _ in range(vector_size)]
            vec2_filter = [0 for _ in range(vector_size)]
            
            for i in range(len(vec1)):
                vec1_filter[int(vec1[i]) % vector_size] = 1
            
            for i in range(len(vec2)):
                vec2_filter[int(vec2[i]) % vector_size] = 1

            # Add them to the new batch
            vectors.append([vec1_filter, vec2_filter])

    new_batch = [torch.Tensor(vectors), torch.Tensor(labels)]

    return new_batch

# def reset_weights(m):
#     for layer in m.children():
#         if isinstance(layer, torch.nn.Sequential):
#             for sub_layer in layer:
#                 if hasattr(sub_layer, "reset_parameters"):
#                     sub_layer.reset_parameters()
#         elif hasattr(layer, "reset_parameters"):
#             layer.reset_parameters()
#         else:
#             print("No parameters to reset for this layer")


def access_data(bloom_filter_size):
    vector_1 = list()
    vector_2 = list()

    with open("../data/0") as f:
        for line in f.readlines():
            # pair = []
            units = line.split("\t")

            for i in range(1, len(units)):
                string_nums = re.split(", |\[|\]", units[i])
                nums = []
                for j in range(1, len(string_nums) - 1):
                    nums.append(int(string_nums[j]))

                if i == 1:
                    vector_1.append(nums)
                else:
                    vector_2.append(nums)

    return vector_1, vector_2


if __name__ == "__main__":
    bloom_filter_size = 2048
    numLayers = 2
    hash_bit = 16
    layer_size = 256

    vector_1, vector_2 = access_data(bloom_filter_size)

    # Compile the dataset into a list
    dataset = []
    for i in range(len(vector_1)):
        dataset.append([[vector_1[i], vector_2[i]]])

    # Simple train/test split of the data
    total_count = len(dataset)
    train_count = int(0.8 * total_count)
    test_count = total_count - train_count
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_count, test_count)
    )

    print(len(train_dataset))
    print(len(test_dataset))

    print("Train/test split completed.")

    # initialize the model
    model = DocumentNetFc(bloom_filter_size, numLayers, hash_bit, layer_size)
    writer = SummaryWriter()
    print(model)

    # initialize the optimizer
    num_epochs = 5
    learning_rate = 1e-5
    momentum = 0.9
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)

    batch_size = 64
    pct = .25

    for epoch in range(num_epochs):
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn_batch(batch, pct, bloom_filter_size))
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs1 = inputs.select(1, 0)
            inputs2 = inputs.select(1, 1)

            # Training pass
            optimizer.zero_grad()

            # Perform forward pass
            outputs1 = model(torch.Tensor(inputs1))
            outputs2 = model(torch.Tensor(inputs2))

            # Get the loss from the network outputs
            loss_value = loss.pairwise_loss_updated(outputs1, outputs2, targets)
            writer.add_scalar(tag="Loss/train", scalar_value=loss_value, global_step=epoch)

            # Compute the gradients
            loss_value.backward()

            # Update the parameters
            optimizer.step()

            running_loss += loss_value.item()
        else:
            print(f"Training loss: {running_loss / len(train_loader)}")

    writer.flush()
    writer.close()
    print("Training is successfully completed.")

    # test the neural network
    test_loss = 0.0
    num_correct = 0
    num_total = 0

    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn_batch(batch, pct, bloom_filter_size))

    for inputs, targets in testloader:
        inputs1 = inputs.select(1, 0)
        inputs2 = inputs.select(1, 1)

        outputs1 = model(inputs1)
        outputs2 = model(inputs2)

        for i in range(len(outputs1)):
            if (targets[i] == 1. and (outputs1[i] == outputs2[i]).all()) or (
                    targets[i] == 0. and not (outputs1[i] == outputs2[i]).all()):
                num_correct += 1
            num_total += 1

    print(num_correct)
    print(num_total)
    
    print("The testing accuracy is:", (num_correct / num_total))
