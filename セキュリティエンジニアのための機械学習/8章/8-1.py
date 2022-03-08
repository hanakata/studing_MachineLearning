import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# ラッパーおよびユーティリティをインポートする
from art.estimators.classification.keras import KerasClassifier
from art.utils import load_mnist

# MNISTデータセットをロードする
(X_train, y_train), (X_test, y_test), \
    min_pixel_value, max_pixel_value = load_mnist()

nb_classes=10

# 攻撃対象のモデルを定義する
model = Sequential()
model.add(Conv2D(1,kernel_size=(7, 7), activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])

victim_classifier = KerasClassifier(model,
                                    clip_values=(0, 1), 
                                    use_logits=False)
victim_classifier.fit(X_train, y_train, nb_epochs=5, batch_size=128)

# 窃取先のモデルの雛形を定義する
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.01),
              metrics=['accuracy'])

thieved_classifier = KerasClassifier(model,
                                     clip_values=(0, 1), 
                                     use_logits=False)

# 攻撃手法をインポートする
from art.attacks.extraction.copycat_cnn import CopycatCNN

attack = CopycatCNN(classifier=victim_classifier,
                    batch_size_fit=16,
                    batch_size_query=16,
                    nb_epochs=10,
                    nb_stolen=1000)

# 攻撃結果として訓練済のサロゲートモデルを得る
thieved_classifier = attack.extract(x=X_train,
                                    thieved_classifier=thieved_classifier)

# 結果を表示する
victim_preds = np.argmax(victim_classifier.predict(x=X_train[:100]), 
                         axis=1)
thieved_preds = np.argmax(thieved_classifier.predict(x=X_train[:100]),
                          axis=1)
acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
print('Accuracy of the surrogate model: {}%'.format(acc * 100))