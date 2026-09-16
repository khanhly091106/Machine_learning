import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Du lieu chieu cao va can nang
X = np.array([147,150,153,155,158,160,163,165,168,170,173,175,178,180,183])
y = np.array([49,50,51,52,54,56,58,59,60,72,63,64,66,67,68])

# Chia du lieu thanh train va test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Tao mo hinh cay quyet dinh
# Cay phat trien den khi qua sat du lieu train
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train.reshape(-1, 1), y_train)

# Du doan
train_pred = model.predict(X_train.reshape(-1, 1))
test_pred = model.predict(X_test.reshape(-1, 1))

# Tinh loi
train_error = mean_squared_error(y_train, train_pred)
test_error = mean_squared_error(y_test, test_pred)

print("Training Error:", train_error)
print("Testing Error :", test_error)

if train_error < test_error:
    print("Mo hinh co dau hieu Overfitting")
