import random
for line in open("data.txt","r"):
    if random.random() < 0.67:
        with open('data_train.txt', 'a') as f:
            f.write(line)
    else:
        with open('data_val.txt', 'a') as f:
            f.write(line)
