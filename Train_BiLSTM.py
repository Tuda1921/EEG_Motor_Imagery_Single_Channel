import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.optimizers import Adam

# Đọc dữ liệu từ file CSV
df = pd.read_csv('dataC3.csv')
df = df[df['Label'] != 0]
# Với cột cuối là nhãn
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding cho nhãn
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

# Tạo thư mục để lưu các plot và mô hình
model_dir = 'models_traiphai_new'
os.makedirs(model_dir, exist_ok=True)

# Danh sách để lưu tên file của các mô hình đã train
model_filenames = []

# Tạo DataFrame để lưu history của loss và accuracy
history_df = pd.DataFrame(columns=['fold', 'epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'])

for fold_index, (train_index, test_index) in enumerate(kfold.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Chia thành tập huấn luyện và tập kiểm tra
    time_steps = 80
    n_features = X.shape[1]

    # Chuẩn bị đầu vào và đầu ra cho mô hình Bi-LSTM
    def prepare_data(data, labels, time_steps):
        X = []
        y = []
        for i in range(len(data) - time_steps):
            X.append(data[i:i+time_steps])
            y.append(labels[i+time_steps])
        return np.array(X), np.array(y)
    X_train, y_train = prepare_data(X_train, y_train, time_steps)
    X_test, y_test = prepare_data(X_test, y_test, time_steps)

    # Xây dựng mô hình Bi-LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(units=16, activation='relu'), input_shape=(time_steps, n_features)))
    # model.add(Bidirectional(LSTM(units=16, activation='relu'), input_shape=(time_steps, n_features)))
    print(model.output_shape)
    model.add(Dropout(0.1))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.1)) #0.2
    model.add(Dense(units=2, activation='softmax'))
    #learning rate
    lr = 0.0001
    optimizer = Adam(learning_rate=lr)

    # Biên dịch mô hình
    model.compile(optimizer=optimizer , loss='categorical_crossentropy', metrics=['accuracy'])

    # Đặt callbacks để dừng sau 10 epochs
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)

    # Huấn luyện mô hình và lưu history
    history = model.fit(X_train, y_train, epochs=200, batch_size=96, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Lưu mô hình
    model_filename = f"BiLSTM_Brainwave_{fold_index}_accuracy_{history.history['val_accuracy'][-1]*100:.2f}.h5"
    model.save(os.path.join(model_dir, model_filename))
    model_filenames.append(model_filename)

    # Đánh giá mô hình trên tập kiểm tra
    _, accuracy = model.evaluate(X_test, y_test)
    fold_scores.append(accuracy)

    # Vẽ đồ thị loss và accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(f"{model_dir}/{os.path.splitext(model_filename)[0]}_loss.png")
    plt.clf()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig(f"{model_dir}/{os.path.splitext(model_filename)[0]}_accuracy.png")
    plt.clf()

    # Lưu accuracy và loss của fold hiện tại vào DataFrame
    for epoch, (train_loss, test_loss, train_accuracy, test_accuracy) in enumerate(zip(
        history.history['loss'],
        history.history['val_loss'],
        history.history['accuracy'],
        history.history['val_accuracy']
    ), 1):
        history_df = pd.concat([history_df, pd.DataFrame({
            'fold': [fold_index],
            'epoch': [epoch],
            'train_loss': [train_loss],
            'test_loss': [test_loss],
            'train_accuracy': [train_accuracy],
            'test_accuracy': [test_accuracy]
        })], ignore_index=True)

    # Lưu thông tin của fold hiện tại vào file txt
    with open(f'{model_dir}/{os.path.splitext(model_filename)[0]}.txt', 'w') as file:
        file.write(f"Train Loss: {history.history['loss'][-1]}\n")
        file.write(f"Train Accuracy: {history.history['accuracy'][-1]}\n")
        file.write(f"Validation Loss: {history.history['val_loss'][-1]}\n")
        file.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]}\n")

# In kết quả
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: Accuracy = {score}")

print("Average Accuracy:", np.mean(fold_scores))

# Lưu DataFrame vào file CSV
history_df.to_csv('history_bi_lstm.csv', index=False)
