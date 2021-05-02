import random

data_size = 80000

# random.seed(5)

sequence = []

# # repeat same move random number of times
for i in range(data_size):
    num = random.randint(0,2)
    for j in range(random.randint(1,10)):
        if num == 0:
            sequence.append('r')
        elif num == 1:
            sequence.append('p')
        else:
            sequence.append('s')


# repeat sequence random number of times
# for i in range(data_size):
#     seq = []
#     for j in range(random.randint(1, 20)):
#         num = random.randint(0,2)
#         if num == 0:
#             seq.append('r')
#         elif num == 1:
#             seq.append('p')
#         else:
#             seq.append('s')
#
#     rep = random.randint(1, 10)
#     for k in range(rep):
#         sequence.extend(seq)


# pure rng
# for i in range(data_size):
#     num = random.randint(0,2)
#     if num == 0:
#         sequence.append('r')
#     elif num == 1:
#         sequence.append('p')
#     else:
#         sequence.append('s')

f = open("rps_data.txt", "w")
for item in sequence:
    f.write(item)
f.close()