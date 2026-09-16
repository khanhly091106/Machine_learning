import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Du lieu
X = np.array([147,150,153,155,158,160,163,165,168,170,173,175,178,180,183])
y = np.array([49,50,51,52,54,56,58,59,60,72,63,64,66,67,68])

# Chia du lieu thanh 5 phan
kf = KFold(n_splits=5, shuffle=True, random_state=42)

errors = []

# Lap 5 lan
for i, (train, validation) in enumerate(kf.split(X), 1):

    # Lay du lieu train
    X_train = X[train]
    y_train = y[train]
    # Lay du lieu validation
    X_val = X[validation]
    y_val = y[validation]
    # Tao mo hinh
    model = DecisionTreeRegressor(random_state=42)
    # Hoc tu tap train
    model.fit(X_train.reshape(-1, 1), y_train)
    # Du doan tap validation
    y_pred = model.predict(X_val.reshape(-1, 1))
    # Tinh loi
    error = mean_squared_error(y_val, y_pred)
    errors.append(error)
    print("Lan", i)
    print("Validation Error =", error)
    print()

# Tinh loi trung binh
cv_error = np.mean(errors)
print("---------------------------")
print("CV Error =", cv_error)
