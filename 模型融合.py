import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.utils import shuffle
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score

from sklearn.multioutput import MultiOutputRegressor


params = {
        'booster': 'gbtree',
        # 'objective': 'multi:softmax',  # 多分类的问题、
        # 'objective': 'multi:softprob',   # 多分类概率
        #     'objective': 'reg:linear',
        'eval_metric': 'mae',
        # 'num_class': 9,  # 类别数，与 multisoftmax 并用
        #     'gamma': 0,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        #     'max_depth': 5,  # 构建树的深度，越大越容易过拟合
        'max_depth': 5,  # 构建树的深度，越大越容易过拟合
        'alpha': 0,  # L1正则化系数
        'lambda': 15,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.6,  # 随机采样训练样本
        'colsample_bytree': 0.5,  # 生成树时进行的列采样
        'min_child_weight': 5,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        #     'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        #     'eta': 0.15,  # 如同学习率
        'eta': 0.1,  # 如同学习率
        'seed': 0,
        'nthread': -1,  # cpu 线程数
        # 'tree_method': 'gpu_hist',
        'missing': 1,

        'n_estimators': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,

        #     'num_boost_round':500
        # 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)
        # 'metric': 'auc'
    }


df = pd.read_excel('A_DG_GEN_CURVE_DAY.xlsx',engine='openpyxl')
df.replace('None', 0, inplace=True)
df.replace(np.nan, 0, inplace=True)
df.fillna(method = 'ffill', axis = 0)

cons_no_s=df['CUST_NO'].unique().tolist()
df0=df[df['CUST_NO']==cons_no_s[0]]
df1=df[df['CUST_NO']==cons_no_s[1]]
df2=df[df['CUST_NO']==cons_no_s[2]]
df3=df[df['CUST_NO']==cons_no_s[3]]
df4=df[df['CUST_NO']==cons_no_s[4]]
df = df1.iloc[:,16:]

#构建特征和标签
n_vars = 1 if type(df) is list else df.shape[1]

df = DataFrame(df)
cols, names = list(), list()
# input sequence (t-n, ... t-1)
for i in range(1, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
for i in range(0, 1):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
# put it all together
agg = pd.concat(cols, axis=1)
agg.columns = names
# drop rows with NaN values
agg.dropna(inplace=True)

#data partition
feature = agg.iloc[:, 0:24]
label = agg.iloc[:, 24:]


feature, label = shuffle(feature, label, random_state=1337)
training_set_size = int(len(feature) * 0.8)

train_feature = feature.iloc[:training_set_size]
test_feature = feature.iloc[training_set_size:]

train_label = label.iloc[:training_set_size]
test_label = label.iloc[training_set_size:]


model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **params))
model.fit(train_feature,train_label)
result = model.predict(train_feature)
result = DataFrame(result)

result.replace('None', 0, inplace=True)
result.replace(np.nan, 0, inplace=True)
result.fillna(method = 'ffill', axis = 0)

#cons_no_s=df['CUST_NO'].unique().tolist()
#df0=df[df['CUST_NO']==cons_no_s[0]]
#df1=df[df['CUST_NO']==cons_no_s[1]]
#df2=df[df['CUST_NO']==cons_no_s[2]]
#df3=df[df['CUST_NO']==cons_no_s[3]]
df4=df[df['CUST_NO']==cons_no_s[4]]
#df = df1.iloc[:,16:]

#构建特征和标签
n_vars = 1 if type(result) is list else result.shape[1]

df = DataFrame(result)
cols, names = list(), list()
# input sequence (t-n, ... t-1)
for i in range(1, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
for i in range(0, 1):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
# put it all together
agg = pd.concat(cols, axis=1)
agg.columns = names
# drop rows with NaN values
agg.dropna(inplace=True)

data = agg.values
print('data', data.shape)
feature = data[:, 0:96]
label = data[:, 96:]
# scaler = MinMaxScaler()
training_set_size = int(len(feature) * 0.7)
# feature = scaler.fit_transform(feature)

from sklearn.utils import shuffle
feature, label = shuffle(feature, label, random_state=1337)
train_feature = feature[:training_set_size]
test_feature = feature[training_set_size:]
train_label = label[:training_set_size]
test_label = label[training_set_size:]
train_feature = train_feature.reshape((train_feature.shape[0], 1, train_feature.shape[1]))
test_feature = test_feature.reshape((test_feature.shape[0], 1, test_feature.shape[1]))

units = 32
epochs = 500
batch_size = 64
train_x = train_feature.astype('float64')
test_x = test_feature.astype('float64')
train_y = train_label.astype('float64')
test_y = test_label.astype('float64')

model = Sequential()
model.add(LSTM(units,return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(.1))
model.add(LSTM(units))
model.add(Dropout(.1))
model.add(Dense(test_y.shape[1]))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# print(model.summary())

# 设置学习
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                              epsilon=0.0001, cooldown=0, min_lr=0)

# fit network
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                    validation_data=(test_x, test_y), verbose=2, shuffle=False, callbacks=[reduce_lr])

y_predicted = model.predict(test_x)


y_predicted.replace('None', 0, inplace=True)
y_predicted.replace(np.nan, 0, inplace=True)
y_predicted.fillna(method = 'ffill', axis = 0)

#构建特征和标签
n_vars = 1 if type(y_predicted) is list else y_predicted.shape[1]

df = DataFrame(y_predicted)
cols, names = list(), list()
# input sequence (t-n, ... t-1)
for i in range(1, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
for i in range(0, 1):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
# put it all together
agg = pd.concat(cols, axis=1)
agg.columns = names
# drop rows with NaN values
agg.dropna(inplace=True)

#data partition
feature = agg.iloc[:, 0:24]
label = agg.iloc[:, 24:]


feature, label = shuffle(feature, label, random_state=1337)
training_set_size = int(len(feature) * 0.8)

train_feature = feature.iloc[:training_set_size]
test_feature = feature.iloc[training_set_size:]

train_label = label.iloc[:training_set_size]
test_label = label.iloc[training_set_size:]


model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **params))
model.fit(train_feature,train_label)
result = model.predict(train_feature)
result1 = DataFrame(result)



