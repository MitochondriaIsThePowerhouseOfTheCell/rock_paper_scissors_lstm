import matplotlib.pyplot as plt
import csv

look_back = [i for i in range(10, 105, 5)]
wins, ties, loss = [], [], []
with open('look_back_cv.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        wins.append([float(i) for i in row])
        # ties = [float(i) for i in csv_reader[0]]
        # loss = [float(i) for i in csv_reader[0]]

# print(look_back)
# print(len(wins[0]))
# print(wins[1])
# print(wins[2])

plt.title('AI win-tie-loss rates over range of look-back steps')
plt.plot(look_back, wins[0], color='blue')
plt.plot(look_back, wins[1], color='yellow')
plt.plot(look_back, wins[2], color='red')
plt.ylabel('Rate of Occurrence')
plt.xlabel('Look-back Steps')
plt.legend(['Win rate', 'Tie rate', 'Loss rate'], loc='upper right')
plt.show()
