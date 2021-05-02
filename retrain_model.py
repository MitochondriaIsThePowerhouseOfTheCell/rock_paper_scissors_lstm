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
output_dim = 3
input_dim = 3
look_back = 25
train_pct = 0.9
dropout_pct = 0.3
epochs = 25
num_nodes = 10
validation_split = 0.2
batch_size = 50

plot_pct = 0.01

now = time.strftime("%y%m%d%H%M%S")


# ----------------------------- main program ----------------------------
raw_data = load_data("rps_data.txt")

print('Input length: ', len(raw_data))
print('Train %: ', train_pct * 100)
print('Sequence length: ', look_back)

# split train/test sets
trans_data = embedding_layer(raw_data)
train_size = int(len(trans_data) * train_pct)
train = trans_data[0:train_size]
# print(train)
test = trans_data[train_size:len(trans_data)]
print('Train set size:', train_size)
print('Test set size :', len(trans_data) - train_size)

# make sequential data
trainX, trainY = sequence_maker(train, look_back)
testX, testY = sequence_maker(test, look_back)


# --------------------------------- make model ---------------------------------------
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/' + now, histogram_freq=1, write_graph=True, write_grads=True,
                                         write_images=False)

model = keras.models.load_model('1_mil_model.tf')
model.summary()

hist = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split,
                 callbacks=[callback, tbCallBack])

model.save('TFotR_retrained.tf')


# ----------------------- plot loss per epoch ------------------------
fig = plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], color='#785ef0')
plt.plot(hist.history['val_loss'], color='#dc267f')
plt.title('Model Loss Progress')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='upper right')
plt.show()


# -------------------------------- make predictions -----------------------------
testPredict = model.predict(testX)
# print(testPredict)

testPredictTrans = np.zeros(testPredict.shape, dtype=np.int)
for index, i in enumerate(testPredict.argmax(axis=1)):
    testPredictTrans[index, i] = 1

PredictResult, PredictShape = testPredictTrans, testPredict.shape[0]
truth = testY
# print(testPredictTrans)
# print(truth)


# ------------------- check prediction correctness -------------------------
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


# ------------------------ calculate win/tie/loss rate -----------------------------------
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
rounds = [i for i in range(0, testPredict.shape[0]+1, int(1/plot_pct))]

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
