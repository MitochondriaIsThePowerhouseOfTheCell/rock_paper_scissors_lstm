import numpy as np
import keras
import matplotlib.pyplot as plt
import time


# ----------------------------- functions --------------------------
def load_data(filename):
    f = open(filename, 'r')
    dat = f.read()
    f.close()
    return dat


def embedding_layer(data):
    trsl = []
    for i in range(len(data)):
        if data[i] == 'r':
            trsl.append([1,0,0])
        elif data[i] == 'p':
            trsl.append([0,1,0])
        else:
            trsl.append([0,0,1])
    trsl_array = np.array(trsl)
    # print(trsl_array)
    return trsl_array


def sequence_maker(rpsdata, look_back):
    dataX, dataY = [], []
    numseq = len(rpsdata) - look_back
    for i in range(numseq-1):
        dataX.append(rpsdata[i:(i+look_back), :])
        dataY.append(rpsdata[i+look_back, :])
    return np.array(dataX), np.array(dataY)


# ---------------------------- param config ----------------------------
look_back = 25
test_pct = 1

plot_pct = 0.001

now = time.strftime("%y%m%d%H%M%S")


# ----------------------------- main program ----------------------------
raw_data = load_data("converted_TTT.txt")
print('Input length: ', len(raw_data))
print('Sequence length: ', look_back)

# split test set
train_size = int(len(raw_data) * (1-test_pct))
test = raw_data[train_size:len(raw_data)]

trans_data = embedding_layer(test)

# make sequential data
testX, testY = sequence_maker(trans_data, look_back)

print('Test set shape:', testX.shape)
print('Test label shape:', testY.shape)


# -------------------------------- make model --------------------------------------

model = keras.models.load_model('TFotR_retrained.tf')
model.summary()

# make predictions
testPredict = model.predict(testX)
# print(testPredict)

print("Done predicting")

testPredictTrans = np.zeros(testPredict.shape, dtype=np.int)
for index, i in enumerate(testPredict.argmax(axis=1)):
    testPredictTrans[index, i] = 1

PredictResult, PredictShape = testPredictTrans, testPredict.shape[0]
truth = testY
# print(testPredictTrans)
# print(truth)


# see if prediction match
# record who win each round
AI_win = []
for index, predict in enumerate(PredictResult):
    predict_index = predict.argmax(axis=0)
    AI_move = (predict_index + 1) % 3  # calculate the AI opposing move
    if predict_index == truth[index].argmax(axis=0):  # if prediction is correct
        AI_win.append(1)  # win
    elif AI_move == truth[index].argmax(axis=0):  # if move is same as opponent
        AI_win.append(0)  # tie
    else:
        AI_win.append(-1)  # lose

print("Done recording win")


# ------------------------------ calculate win/tie/loss rate ------------------------------------
AI_win_rate, AI_tie_rate, AI_loss_rate = [], [], []
cum_win, cum_tie, cum_loss = 0, 0, 0
for index, i in enumerate(AI_win):
    if AI_win[index] == 1:
        cum_win += 1
    elif AI_win[index] == 0:
        cum_tie += 1
    else:
        cum_loss += 1
    AI_win_rate.append(cum_win / (index + 1))
    AI_tie_rate.append(cum_tie / (index + 1))
    AI_loss_rate.append(cum_loss / (index + 1))

print("Done calculating win rate")


# -------------------------------------- result ------------------------------------------------
rounds = [i for i in range(0, testPredict.shape[0], int(1/plot_pct))]

# plot win/tie/loss rate
plt.title('AI win rates over number of games')
plt.plot(rounds, AI_win_rate[::int(1/plot_pct)], color='blue')
plt.plot(rounds, AI_tie_rate[::int(1/plot_pct)], color='yellow')
plt.plot(rounds, AI_loss_rate[::int(1/plot_pct)], color='red')
plt.ylabel('Rate of Occurrence')
plt.xlabel('Number of Games')
plt.legend(['Win rate', 'Tie rate', 'Loss rate'], loc='upper right')
plt.show()

print("AI Win Rate: ", AI_win_rate[-1]*100, "%")
print("AI Tie Rate: ", AI_tie_rate[-1]*100, "%")
print("AI Loss Rate: ", AI_loss_rate[-1]*100, "%")
