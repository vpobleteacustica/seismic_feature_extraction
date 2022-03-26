from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import *
from tensorflow.python.keras import backend as K
import tensorflow as tf

def CNN_CT(ceps_train, dist_train, ceps_val, dist_val):
    model  = Sequential() # modelo creado 
    # Bloque convolucional CNN
    model.add(Conv2D(64, kernel_size = (4,1), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (4,1))) # (None, 512, 702, 64) para mantener la relación 512 a 701. O más chiquitito...
    model.add(Dropout(0.15))

    # Bloque convolucional CNN
    model.add(Conv2D(64, kernel_size = (1,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (1,6))) # (None, 512, 702, 64) para mantener la relación 512 a 701. O más chiquitito...
    model.add(Dropout(0.15))
    
    # CNN -> FC
    model.add(Flatten())
    
    # Bloque FC 
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis = -1))
    #model.add(Dropout(0.5))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    
    # Salida
    model.add(Dense(1))
    lr = 1e-5
    opt    = Adam(learning_rate = lr)
    #opt = tf.keras.optimizers.RMSprop(lr, clipnorm = 1.0, centered = True, momentum = 0.85, rho = 0.65, epsilon=0.00000001) #RMSprop(lr, name="RMSprop", clipnorm=0.9, decay = 0.1, clipvalue = 1.0) 
    #opt = RMSprop(lr, clipnorm = 1.0, centered = True, momentum = 0.85, rho = 0.65, epsilon=0.00000001) #RMSprop(lr, name="RMSprop", clipnorm=0.9, decay = 0.1, clipvalue = 1.0)
    #model.compile(loss = 'mse', optimizer = opt)
    model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 250,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    #ceps_train = ceps_train.reshape(ceps_train.shape[1], ceps_train.shape[2], ceps_train.shape[0])
    #ceps_val = ceps_val.reshape(ceps_val.shape[1], ceps_val.shape[2], ceps_val.shape[0])
    
    hist = model.fit(x = ceps_train, y = dist_train, batch_size = 1, validation_data = (ceps_val, dist_val), epochs=1000, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CNN_DISTANCE(ceps_train, dist_train, ceps_val, dist_val, max_epoch, lr):
    model  = Sequential() # modelo creado 
    # Bloque convolucional CNN
    model.add(Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    #model.add(Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (4,6))) # (None, 512, 702, 64) para mantener la relación 512 a 701. O más chiquitito...
    model.add(Dropout(0.15))
    
    # CNN -> FC
    model.add(Flatten())
    
    # Bloque FC 
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis = -1))
    #model.add(Dropout(0.5))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    
    # Salida
    model.add(Dense(1))

    #lr = 1e-4 # aprendizaje para magnitude
    #lr = 1e-6 # aprendizaje para distance
    opt    = Adam(learning_rate = lr)
    #opt = tf.keras.optimizers.RMSprop(lr, clipnorm = 1.0, centered = True, momentum = 0.85, rho = 0.65, epsilon=0.00000001) #RMSprop(lr, name="RMSprop", clipnorm=0.9, decay = 0.1, clipvalue = 1.0) 
    #opt = RMSprop(lr, clipnorm = 1.0, centered = True, momentum = 0.85, rho = 0.65, epsilon=0.00000001) #RMSprop(lr, name="RMSprop", clipnorm=0.9, decay = 0.1, clipvalue = 1.0)
    model.compile(loss = 'mse', optimizer = opt)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    #ceps_train = ceps_train.reshape(ceps_train.shape[1], ceps_train.shape[2], ceps_train.shape[0])
    #ceps_val = ceps_val.reshape(ceps_val.shape[1], ceps_val.shape[2], ceps_val.shape[0])
    
    hist = model.fit(x = ceps_train, y = dist_train, batch_size = 1, validation_data = (ceps_val, dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CNN(ceps_train, dist_train, ceps_val, dist_val, max_epoch, lr):
    model  = Sequential() # modelo creado 
    # Bloque convolucional CNN
    model.add(Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    #model.add(Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (4,6))) # (None, 512, 702, 64) para mantener la relación 512 a 701. O más chiquitito...
    model.add(Dropout(0.15))
    
    # CNN -> FC
    model.add(Flatten())
    
    # Bloque FC 
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis = -1))
    #model.add(Dropout(0.5))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    
    # Salida
    model.add(Dense(1))


    '''
    model  = Sequential() # modelo creado 
    # Bloque convolucional CNN
    model.add(Conv2D(64, kernel_size = (4,3), padding = 'same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (4,3)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size = (2,3), padding = 'same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (2,1)))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, kernel_size = (3,3), padding = 'same'))
    model.add(Activation('relu'))
    ##model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (2,1)))
    model.add(Dropout(0.05))

    model.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
    model.add(Activation('relu'))
    ##model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (2,1)))
    model.add(Dropout(0.05))

    # CNN -> FC
    model.add(Flatten())
    # Bloque FC 
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis = -1))
    #model.add(Dropout(0.5))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    # Salida
    model.add(Dense(1))  # activación de salida lineal en regresión
    ## DIEGO:.... tasa de aprendizaje, probar dropout por bloque conv; luego apagar primero y probar por speardo, y despues juntos
    '''
    #lr = 1e-4 # aprendizaje para magnitude
    #lr = 1e-6 # aprendizaje para distance
    opt    = Adam(learning_rate = lr)
    #opt = tf.keras.optimizers.RMSprop(lr, clipnorm = 1.0, centered = True, momentum = 0.85, rho = 0.65, epsilon=0.00000001) #RMSprop(lr, name="RMSprop", clipnorm=0.9, decay = 0.1, clipvalue = 1.0) 
    #opt = RMSprop(lr, clipnorm = 1.0, centered = True, momentum = 0.85, rho = 0.65, epsilon=0.00000001) #RMSprop(lr, name="RMSprop", clipnorm=0.9, decay = 0.1, clipvalue = 1.0)
    model.compile(loss = 'mse', optimizer = opt)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    #ceps_train = ceps_train.reshape(ceps_train.shape[1], ceps_train.shape[2], ceps_train.shape[0])
    #ceps_val = ceps_val.reshape(ceps_val.shape[1], ceps_val.shape[2], ceps_val.shape[0])
    
    hist = model.fit(x = ceps_train, y = dist_train, batch_size = 1, validation_data = (ceps_val, dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CNN_SP(ceps_train, sp_train, dist_train, ceps_val, sp_val, dist_val):
    cnn_input = Input(shape=(512, 702, 3))
    sp_input  = Input(shape=(1,))
    
    # Bloque convolucional CNN
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('elu')(cnn)
    cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    #cnn2        = Conv2D(64, kernel_size = (8,1), padding = 'same', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1))(cnn)
    ##cnn2         = BatchNormalization(axis = -1)(cnn2)
    #cnn2        = AveragePooling2D(pool_size = (2,1))(cnn2)
    #cnn2        = Dropout(0.2)(cnn2)
    cnn         = Flatten()(cnn)
    cnn         = Dense(128)(cnn)
    cnn         = Dropout(0.5)(cnn)
    cnn         = Activation('elu')(cnn)
    cnn         = Dense(1)(cnn)
    cnn         = Activation('elu')(cnn)

    sp          = Dense(1)(sp_input)
    sp          = Activation('elu')(sp)
    #sp          = sp_input
    merge       = Concatenate()([cnn, sp])
    #mlp         = Dense(2049)(merge)
    #mlp         = Activation('relu')(mlp)
    #mlp         = Dropout(0.2)(mlp)
    #mlp         = Dense(2048)(mlp)
    #mlp         = Activation('relu')(mlp)
    #mlp         = Dense(128)(mlp)
    #mlp         = Activation('relu')(mlp)
    mlp         = Dense(1)(merge)
    model       = Model(inputs = [cnn_input, sp_input], outputs = mlp)

    lr = 1e-4
    opt    = Adam(learning_rate = lr)
    model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, sp_train], y = dist_train, batch_size = 4, validation_data = ([ceps_val,sp_val], dist_val), epochs=1000, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CRNN(ceps_train, dist_train, ceps_val, dist_val):
    
    model  = Sequential() # modelo creado 
    # Bloque convolucional CNN
    model.add(Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (4,6))) # (None, 512, 702, 64) para mantener la relación 512 a 701. O más chiquitito...
    model.add(Dropout(0.15))

        # Bloque convolucional CNN
    model.add(Conv2D(32, kernel_size = (2,3), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(AveragePooling2D(pool_size = (2,3))) # (None, 512, 702, 64) para mantener la relación 512 a 701. O más chiquitito...
    model.add(Dropout(0.15))
    
    # cnn -> rnn
    model.add(Reshape(((ceps_train.shape[1]//8)*(ceps_train.shape[2]//18), 32)))

    # bloque rnn
    model.add(Bidirectional(GRU(16, return_sequences=True), merge_mode='mul'))
    model.add(Dropout(0.25))
    model.add(Bidirectional(GRU(16, return_sequences=True), merge_mode='mul'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.25))
    #model.add(Bidirectional(GRU(16)))

    # CNN -> FC
    #model.add(Flatten())
    
    # Bloque FC 
    model.add(Dense(32))
    model.add(Activation('relu'))
    
    # Salida
    model.add(Dense(1))

    lr = 1e-6 
    opt    = Adam(learning_rate = lr)
    model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    #ceps_train = ceps_train.reshape(ceps_train.shape[1], ceps_train.shape[2], ceps_train.shape[0])
    #ceps_val = ceps_val.reshape(ceps_val.shape[1], ceps_val.shape[2], ceps_val.shape[0])
    
    hist = model.fit(x = ceps_train, y = dist_train, batch_size = 1, validation_data = (ceps_val, dist_val), epochs=1000, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr 

def CNN_STACKED(ceps_train, delta_train, delta_delta_train, dist_train, ceps_val, delta_val, delta_delta_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    d_input     = Input(shape=(512, 702, 3))
    dd_input    = Input(shape=(512, 702, 3))
    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(128)(cnn)
    cnn         = Activation('relu')(cnn)

    # cnn d
    cnn_d         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(d_input)
    cnn_d         = Activation('relu')(cnn_d)
    cnn_d         = BatchNormalization(axis = -1)(cnn_d)
    cnn_d         = AveragePooling2D(pool_size = (4,6))(cnn_d)
    cnn_d         = Dropout(0.15)(cnn_d)
    cnn_d         = Flatten()(cnn_d)
    cnn_d         = Dense(128)(cnn_d)
    cnn_d         = Activation('relu')(cnn_d)

    # cnn dd
    cnn_dd         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(dd_input)
    cnn_dd         = Activation('relu')(cnn_dd)
    cnn_dd         = BatchNormalization(axis = -1)(cnn_dd)
    cnn_dd         = AveragePooling2D(pool_size = (4,6))(cnn_dd)
    cnn_dd         = Dropout(0.15)(cnn_dd)
    cnn_dd         = Flatten()(cnn_dd)
    cnn_dd         = Dense(128)(cnn_dd)
    cnn_dd         = Activation('relu')(cnn_dd)

    # concat
    merge       = Concatenate()([cnn, cnn_d, cnn_dd])
    merge       = Dense(64, activation = 'relu')(merge)
    out         = Dense(1)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, d_input, dd_input], outputs = out)

    #lr = 1e-6
    opt    = Adam(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(0.2*max_epoch),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, delta_train, delta_delta_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, delta_val, delta_delta_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def D_MLP(feat_train, dist_train, feat_val, dist_val, max_epoch, lr):
    feat_input  = Input(shape=(4,))
    
    # mlp
    mlp = Dense(512, activation = 'sigmoid')(feat_input)
    mlp = Dense(256, activation = 'sigmoid')(mlp)
    mlp = Dense(128, activation = 'sigmoid')(mlp)
    mlp = Dense(64,  activation = 'sigmoid')(mlp)
    out = Dense(2,  activation  = 'linear')(mlp)

    model       = Model(inputs = feat_input, outputs = out)

    #lr = 1e-5
    opt    = Adadelta(learning_rate = lr) # Adam
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = feat_train, y = dist_train, batch_size = 1, validation_data = (feat_val, dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def D_MLP_M(feat_train, dist_train, feat_val, dist_val, max_epoch, lr):
    feat_input  = Input(shape=(6,))
    
    # mlp
    mlp = Dense(2048, activation = 'relu')(feat_input)
    mlp = Dense(1024, activation = 'relu')(mlp)
    mlp = Dense(512, activation = 'relu')(mlp)
    mlp = Dense(256, activation = 'relu')(mlp)
    mlp = Dense(128, activation = 'relu')(mlp)
    mlp = Dense(64,  activation = 'relu')(mlp)
    mlp = Dense(16,  activation = 'relu')(mlp)
    out = Dense(2,  activation  = 'linear')(mlp)

    model       = Model(inputs = feat_input, outputs = out)

    #lr = 1e-5
    opt    = Adam(learning_rate = lr) # Adam Adadelta
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100,
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = feat_train, y = dist_train, batch_size = 1, validation_data = (feat_val, dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def D_CNN_MLP_M(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    feat_input  = Input(shape=(6,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(32)(cnn) # fully connected
    cnn         = Activation('relu')(cnn)

    # mlp
    mlp = Dense(1024, activation = 'relu')(feat_input)
    mlp = Dense(512, activation = 'relu')(mlp)
    mlp = Dense(256, activation = 'relu')(mlp)
    mlp = Dense(64, activation = 'relu')(mlp)
    
    # concat
    merge       = Concatenate()([cnn, mlp])
    merge       = Dense(64, activation = 'relu')(merge)
    out         = Dense(2)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = SGD(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def D_CNN_MLP_M_NSP(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    feat_input  = Input(shape=(6,))
    #feat_input  = Input(shape=(5,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(32)(cnn) # fully connected
    cnn         = Activation('relu')(cnn)

    # mlp
    mlp = Dense(1024, activation = 'relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.00001))(feat_input)
    #mlp = Dense(512, activation = 'relu')(feat_input) l1_l2(l1=1e-5, l2=1e-4)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(64, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.0055), activity_regularizer=l2(0.0000000001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(32, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.000000055), activity_regularizer=l2(0.0000000055))(mlp) 
    #mlp = Dropout(0.15)(mlp)            
    mlp = Dense(16,  activation = 'relu', kernel_regularizer=l2(0.001))(mlp)
    mlp = Dense(8,  activation = 'relu', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    
    # concat
    merge       = Concatenate()([cnn, mlp])
    merge       = Dense(2048, activation = 'relu')(merge)
    merge       = Dense(512, activation = 'relu' )(merge)
    #merge       = Dense(128, activation = 'relu')(merge)
    merge       = Dense(64, activation = 'relu'  )(merge)
    #merge       = Dense(32, activation = 'relu')(merge)
    out         = Dense(2)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = SGD(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CNN_M(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    #feat_input  = Input(shape=(6,))
    feat_input  = Input(shape=(5,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('linear')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(32)(cnn) # fully connected
    cnn         = Activation('linear')(cnn)

    # mlp
    mlp = Dense(1024, activation = 'linear', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.00001))(feat_input)
    #mlp = Dense(512, activation = 'relu')(feat_input) l1_l2(l1=1e-5, l2=1e-4)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(64, activation = 'linear', bias_regularizer=l2(1), kernel_regularizer=l1(0.0055), activity_regularizer=l2(0.0000000001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    #mlp = Dense(32, activation = 'linear', bias_regularizer=l2(1), kernel_regularizer=l1(0.000000055), activity_regularizer=l2(0.0000000055))(mlp) 
    #mlp = Dropout(0.15)(mlp)            
    #mlp = Dense(16,  activation = 'linear', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dense(8,  activation = 'linear', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    
    # concat
    merge       = Concatenate()([cnn, mlp])
    #merge       = Dense(2048, activation = 'linear')(merge)
    #merge       = Dense(512, activation =  'linear' )(merge)
    merge       = Dense(256, activation = 'relu')(merge)
    #merge       = Dense(128, activation = 'relu')(merge)
    #merge       = Dense(64, activation = 'linear'  )(merge)
    #merge       = Dense(32, activation = 'relu')(merge)
    out         = Dense(1)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = SGD(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CNN_NE(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(70100, 2))
    feat_input  = Input(shape=(6,))
    #feat_input  = Input(shape=(5,))

    # cnn
    cnn         = Conv1D(64, kernel_size = (4), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    #cnn         = Conv1D(64, kernel_size = (4), padding = 'same', kernel_regularizer=l2(0.001), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling1D(pool_size = (4))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(32)(cnn) # fully connected
    cnn         = Activation('relu')(cnn)

    # mlp
    mlp = Dense(1024, activation = 'relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.00001))(feat_input)
    mlp = Dense(64, activation = 'relu')(feat_input) #l1_l2(l1=1e-5, l2=1e-4)
    #mlp = Dropout(0.15)(mlp)
    #mlp = Dense(64, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.0055), activity_regularizer=l2(0.0000000001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    #mlp = Dense(32, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.000000055), activity_regularizer=l2(0.0000000055))(mlp) 
    #mlp = Dropout(0.15)(mlp)            
    #mlp = Dense(16,  activation = 'relu', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dense(8,  activation = 'relu', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    
    # concat
    merge       = Concatenate()([cnn, mlp])
    merge       = Dense(2048, activation = 'relu')(merge)
    merge       = Dense(512, activation = 'relu' )(merge)
    #merge       = Dense(128, activation = 'relu')(merge)
    merge       = Dense(64, activation = 'relu'  )(merge)
    #merge       = Dense(32, activation = 'relu')(merge)
    out         = Dense(2)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = SGD(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CRNN_DIST2(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    feat_input  = Input(shape=(6,))
    #feat_input  = Input(shape=(5,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)

    #rnn
    cnn         = Reshape((8192, 117))(cnn)
    cnn         = Permute((2,1))(cnn)
    rnn         = LSTM(256, return_sequences = True)(cnn)
    rnn         = Dropout(0.15)(rnn)

    crnn        = Flatten()(rnn)
    crnn        = Dense(32)(crnn) # fully connected
    crnn        = Activation('relu')(crnn)

    # mlp
    mlp = Dense(1024, activation = 'relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.00001))(feat_input)
    #mlp = Dense(512, activation = 'relu')(feat_input) l1_l2(l1=1e-5, l2=1e-4)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(64, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.0055), activity_regularizer=l2(0.0000000001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(32, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.000000055), activity_regularizer=l2(0.0000000055))(mlp) 
    #mlp = Dropout(0.15)(mlp)            
    mlp = Dense(16,  activation = 'relu', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    
    # concat
    merge       = Concatenate()([crnn, mlp])
    merge       = Dense(2048, activation = 'relu')(merge)
    merge       = Dense(512, activation = 'relu' )(merge)
    #merge       = Dense(128, activation = 'relu')(merge)
    merge       = Dense(64, activation = 'relu'  )(merge)
    #merge       = Dense(32, activation = 'relu')(merge)
    out         = Dense(2)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = SGD(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def CRNN_DIST(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    #feat_input  = Input(shape=(6,))
    feat_input  = Input(shape=(5,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    
    # rnn
    rnn         = LSTM(8, return_sequences = True)(cnn)
    rnn         = Dropout(0.15)(rnn)


    crnn        = Flatten()(rnn)
    crnn        = Dense(32)(crnn) # fully connected
    crnn        = Activation('relu')(crnn)

    # mlp
    mlp = Dense(1024, activation = 'relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.00001))(feat_input)
    #mlp = Dense(512, activation = 'relu')(feat_input) l1_l2(l1=1e-5, l2=1e-4)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(64, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.0055), activity_regularizer=l2(0.0000000001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    mlp = Dense(32, activation = 'relu', bias_regularizer=l2(1), kernel_regularizer=l1(0.000000055), activity_regularizer=l2(0.0000000055))(mlp) 
    #mlp = Dropout(0.15)(mlp)            
    mlp = Dense(16,  activation = 'relu', kernel_regularizer=l2(0.001))(mlp)
    #mlp = Dropout(0.15)(mlp)
    
    # concat
    merge       = Concatenate()([crnn, mlp])
    merge       = Dense(2048, activation = 'relu')(merge)
    merge       = Dense(512, activation = 'relu' )(merge)
    #merge       = Dense(128, activation = 'relu')(merge)
    merge       = Dense(64, activation = 'relu'  )(merge)
    #merge       = Dense(32, activation = 'relu')(merge)
    out         = Dense(2)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = SGD(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def LL2D(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    feat_input  = Input(shape=(6,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(128)(cnn) # fully connected
    cnn         = Activation('relu')(cnn)

    # mlp
    mlp = Dense(1024, activation = 'relu')(feat_input)
    mlp = Dense(512, activation = 'relu')(mlp)
    mlp = Dense(256, activation = 'relu')(mlp)
    mlp = Dense(64,  activation = 'relu')(mlp)
    
    # concat
    merge       = Concatenate()([cnn, mlp])
    merge       = Dense(64, activation = 'relu')(merge)
    out         = Dense(1)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = Adam(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr

def D_CNN_MLP_M_DT(ceps_train, feat_train, dist_train, ceps_val, feat_val, dist_val, max_epoch, lr):
    # inputs
    cnn_input   = Input(shape=(512, 702, 3))
    feat_input  = Input(shape=(7,))

    # cnn
    cnn         = Conv2D(64, kernel_size = (4,6), padding = 'same', kernel_regularizer=l2(1), bias_regularizer=l2(1))(cnn_input)
    cnn         = Activation('relu')(cnn)
    #cnn         = BatchNormalization(axis = -1)(cnn)
    cnn         = AveragePooling2D(pool_size = (4,6))(cnn)
    cnn         = Dropout(0.15)(cnn)
    cnn         = Flatten()(cnn)
    cnn         = Dense(32)(cnn) # fully connected
    cnn         = Activation('relu')(cnn)

    # mlp
    mlp = Dense(1024, activation = 'relu')(feat_input)
    mlp = Dense(64, activation = 'relu')(mlp)
    mlp = Dense(32, activation = 'relu')(mlp)
    mlp = Dense(16,  activation = 'relu')(mlp)
    
    # concat
    merge       = Concatenate()([cnn, mlp])
    merge       = Dense(128, activation = 'relu')(merge)
    merge       = Dense(64, activation = 'relu')(merge)
    out         = Dense(2)(merge) # se conoce como la capa de salida!

    model       = Model(inputs = [cnn_input, feat_input], outputs = out)

    #lr = 1e-6
    opt    = Adam(learning_rate = lr)
    #model.compile(loss = tf.keras.losses.MeanAbsolutePercentageError(), optimizer = opt)
    model.compile(loss = 'mse', optimizer = opt)
    es     = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = int(max_epoch*0.2),
                           restore_best_weights = True) #el mode min dice que cuando el error de validacion pare de decrecer por "patience" epocas se debe detener el entrenamiento
    hist = model.fit(x = [ceps_train, feat_train], y = dist_train, batch_size = 1, validation_data = ([ceps_val, feat_val], dist_val), epochs=max_epoch, callbacks=[es])
    
    print(model.summary())

    loss_train = hist.history['loss']
    loss_val   = hist.history['val_loss']
    return model, loss_train, loss_val, lr