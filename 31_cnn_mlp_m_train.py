import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 
import numpy as np
import tensorflow as tf
import pandas as pd
from distance.utils import *
from distance.models import *
import os.path as path
from time import localtime, strftime
from obspy.geodetics.base import degrees2kilometers

# MARAVILLOSO
tf.random.set_seed(1234)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.InteractiveSession(config=config)

pc        = "kasparov"
nn_type   = "mlp"
max_epoch = 1000
lr        = 1e-6
results_fold = "resultados/"
time         = strftime("%a, %d %b %Y %H:%M:%S", localtime()).replace(", "," ")

path_files_train_val_test = './files_train_val_test/'
csv_folder    = 'mayores4/'  # 'global/'  #'menores4/' 
sub_set_train = '_filtrada' # '_all'
sub_set_test  = '_filtrada' # '_all' 

path_ceps_train    = path_files_train_val_test + 'train' + sub_set_train + 'feature1_mat.npy' 
path_event_train   = path_files_train_val_test + 'train' + sub_set_train + '_param_event.npy'      
path_sp_train      = csv_folder + 'train' + sub_set_train + '.csv' 
path_station_train = path_files_train_val_test + 'train' + sub_set_train + '_param_station.npy'
path_m_train       = path_files_train_val_test + 'train' + sub_set_train + '_m_up_down.npy'
path_dist_train    = path_files_train_val_test + 'train' + sub_set_train + 'distance.npy' 
path_ms_train      = path_files_train_val_test + 'train' + sub_set_train + '_param_m.npy'

path_ceps_val      = path_files_train_val_test + 'val' + sub_set_train + 'feature1_mat.npy'
path_event_val     = path_files_train_val_test + 'val' + sub_set_train + '_param_event.npy'        
path_sp_val        = csv_folder + 'val' + sub_set_train + '.csv' 
path_station_val   = path_files_train_val_test + 'val' + sub_set_train + '_param_station.npy'
path_m_val         = path_files_train_val_test + 'val' + sub_set_train + '_m_up_down.npy'
path_dist_val      = path_files_train_val_test + 'val' + sub_set_train + 'distance.npy'
path_ms_val        = path_files_train_val_test + 'val' + sub_set_train + '_param_m.npy'

path_ceps_test     = path_files_train_val_test + 'test' + sub_set_test + 'feature1_mat.npy'
path_event_test    = path_files_train_val_test + 'test' + sub_set_test + '_param_event.npy'      
path_station_test  = path_files_train_val_test + 'test' + sub_set_test + '_param_station.npy'      
path_sp_test       = csv_folder + 'test' + sub_set_test + '.csv' 
path_m_test        = path_files_train_val_test + 'test' + sub_set_train + '_m_up_down.npy'
path_dist_test     = path_files_train_val_test + 'test' + sub_set_train + 'distance.npy'
path_ms_test       = path_files_train_val_test + 'test' + sub_set_train + '_param_m.npy'

path_n_event_test  = path_files_train_val_test + 'test' + sub_set_test + '_events.txt'      

########################################################

ceps_train       = np.load(path_ceps_train, allow_pickle=True)
event_train      = np.load(path_event_train, allow_pickle=True)
station_train    = np.load(path_station_train, allow_pickle=True)
m_train          = np.load(path_m_train, allow_pickle=True) 
dist_train       = np.load(path_dist_train, allow_pickle=True)
ms_train         = np.load(path_ms_train, allow_pickle=True)
df_sp_train      = pd.read_csv(path_sp_train)
sp_train         = df_sp_train['s_p'].values
#param_train      = tf.convert_to_tensor((sp_train, m_train[:,0], m_train[:,1], m_train[:,2], station_train[:,0], station_train[:,1], station_train[:,2]), dtype = tf.float32)
#param_train      = tf.convert_to_tensor((sp_train, m_train[:,0], m_train[:,1], station_train[:,0], station_train[:,1], station_train[:,2]), dtype = tf.float32)

#param_train      = tf.convert_to_tensor((dist_train, m_train[:,0], m_train[:,1], station_train[:,0], station_train[:,1], station_train[:,2]), dtype = tf.float32)
param_train      = tf.convert_to_tensor((m_train[:,0], m_train[:,1], station_train[:,0], station_train[:,1], station_train[:,2]), dtype = tf.float32)

#param_train      = tf.convert_to_tensor((dist_train, m_train[:,0], m_train[:,1], station_train[:,0], station_train[:,1]), dtype = tf.float32) # sin ID
#param_train      = tf.convert_to_tensor((dist_train, m_train[:,0], m_train[:,1], station_train[:,0], station_train[:,1], station_train[:,2], station_train[:,3]), dtype = tf.float32) # original plus st_gain log10
# param_train      = tf.convert_to_tensor((dist_train, m_train[:,0], m_train[:,1], station_train[:,0], station_train[:,1]), dtype = tf.float32)
param_train      = tf.transpose(param_train)

ceps_val         = np.load(path_ceps_val, allow_pickle=True)
event_val        = np.load(path_event_val, allow_pickle=True)
station_val      = np.load(path_station_val, allow_pickle=True)
m_val            = np.load(path_m_val, allow_pickle=True) 
dist_val         = np.load(path_dist_val, allow_pickle=True)
ms_val           = np.load(path_ms_val, allow_pickle=True)
df_sp_val        = pd.read_csv(path_sp_val)
sp_val           = df_sp_val['s_p'].values
#param_val        = tf.transpose(tf.convert_to_tensor((sp_val, m_val[:,0], m_val[:,1], m_val[:,2], station_val[:,0], station_val[:,1], station_val[:,2]), dtype = tf.float32))  
#param_val        = tf.transpose(tf.convert_to_tensor((sp_val, m_val[:,0], m_val[:,1], station_val[:,0], station_val[:,1], station_val[:,2]), dtype = tf.float32))  

#param_val        = tf.transpose(tf.convert_to_tensor((dist_val, m_val[:,0], m_val[:,1], station_val[:,0], station_val[:,1], station_val[:,2]), dtype = tf.float32))
param_val        = tf.transpose(tf.convert_to_tensor((m_val[:,0], m_val[:,1], station_val[:,0], station_val[:,1], station_val[:,2]), dtype = tf.float32))

#param_val        = tf.transpose(tf.convert_to_tensor((dist_val, m_val[:,0], m_val[:,1], station_val[:,0], station_val[:,1]), dtype = tf.float32)) # sin ID
#param_val        = tf.transpose(tf.convert_to_tensor((dist_val, m_val[:,0], m_val[:,1], station_val[:,0], station_val[:,1], station_val[:,2], station_val[:,3]), dtype = tf.float32)) # original plus st_gain log10 

#param_val        = tf.transpose(tf.convert_to_tensor((dist_val, m_val[:,0], m_val[:,1], station_val[:,0], station_val[:,1]), dtype = tf.float32))  

ceps_test        = np.load(path_ceps_test, allow_pickle=True)
event_test       = np.load(path_event_test, allow_pickle=True)
station_test     = np.load(path_station_test, allow_pickle=True)
ms_test          = np.load(path_ms_test, allow_pickle=True)
m_test           = np.load(path_m_test, allow_pickle=True) 
dist_test        = np.load(path_dist_test, allow_pickle=True)
df_sp_test       = pd.read_csv(path_sp_test)
sp_test          = df_sp_test['s_p'].values
#param_test       = tf.transpose(tf.convert_to_tensor((sp_test, m_test[:,0], m_test[:,1], m_test[:,2], station_test[:,0], station_test[:,1], station_test[:,2]), dtype = tf.float32))  
#param_test       = tf.transpose(tf.convert_to_tensor((sp_test, m_test[:,0], m_test[:,1], station_test[:,0], station_test[:,1], station_test[:,2]), dtype = tf.float32))  

#param_test       = tf.transpose(tf.convert_to_tensor((dist_test, m_test[:,0], m_test[:,1], station_test[:,0], station_test[:,1], station_test[:,2]), dtype = tf.float32))
param_test       = tf.transpose(tf.convert_to_tensor((m_test[:,0], m_test[:,1], station_test[:,0], station_test[:,1], station_test[:,2]), dtype = tf.float32))

#param_test       = tf.transpose(tf.convert_to_tensor((dist_test, m_test[:,0], m_test[:,1], station_test[:,0], station_test[:,1]), dtype = tf.float32)) # sin ID
#param_test       = tf.transpose(tf.convert_to_tensor((dist_test, m_test[:,0], m_test[:,1], station_test[:,0], station_test[:,1], station_test[:,2], station_test[:,3]), dtype = tf.float32)) # original plus st_gain log10
#param_test       = tf.transpose(tf.convert_to_tensor((dist_test, m_test[:,0], m_test[:,1], station_test[:,0], station_test[:,1]), dtype = tf.float32))    
n_event_test_df  = pd.read_csv(path_n_event_test)
n_event_test     = n_event_test_df.values

print('**normalizando features**')

y_train = ms_train
ceps_train, y_train, param_train = unison_shuffled_copies_v3(ceps_train, y_train, param_train)
X_train = tf.reshape(ceps_train,[ceps_train.shape[0], ceps_train.shape[3], ceps_train.shape[2], ceps_train.shape[1]])
X_train = np.abs(X_train)

y_val   = tf.convert_to_tensor(ms_val, dtype=tf.float32)
X_val   = tf.reshape(ceps_val,[ceps_val.shape[0], ceps_val.shape[3], ceps_val.shape[2], ceps_val.shape[1]])
X_val   = np.abs(X_val)

y_test  = ms_test
X_test  = tf.reshape(ceps_test,[ceps_test.shape[0], ceps_test.shape[3], ceps_test.shape[2], ceps_test.shape[1]])
X_test  = np.abs(X_test)

print('**empezando train**')
#model, loss_train, loss_val, lr = D_CNN_MLP_M(X_train, param_train, y_train, X_val, param_val, y_val, max_epoch, lr)
#model, loss_train, loss_val, lr = D_CNN_MLP_M_NSP(X_train, param_train, y_train, X_val, param_val, y_val, max_epoch, lr)
model, loss_train, loss_val, lr = CNN_M(X_train, param_train, y_train, X_val, param_val, y_val, max_epoch, lr)
#model, loss_train, loss_val, lr = CRNN_DIST2(X_train, param_train, y_train, X_val, param_val, y_val, max_epoch, lr)
#model, loss_train, loss_val, lr = D_CNN_MLP_M_DT(X_train, param_train, y_train, X_val, param_val, y_val, max_epoch, lr)

print('**calculando errores**')

er   = []
pred = []
test = []
for t in range(param_test.shape[0]):
    target  = y_test[t]
    param_t = param_test[t, :]
    
    x_test    = X_test[t,:,:,:]
    x_test    = x_test.reshape(1, x_test.shape[0], x_test.shape[1], x_test.shape[2])
    predict = model.predict([x_test, np.array([param_t])])
    #predict = predict.flatten()
    pred.append(predict[0][0])
    test.append(target)
    #print(predict[0])
    #error   = er_pos(target, predict[0])
    error = target - predict[0]
    er.append(error)

pred = np.asarray(pred)
test = np.asarray(test)
#pred = np.around(pred,1)

err_abs = np.mean([abs(test[i]-pred[i]) for i in range(len(pred))], axis=0)
err_rel_abs = [abs( (test[i] - pred[i])/test[i]) for i in range(len(pred))]
err_rel = 100*np.mean(err_rel_abs, axis=0)
err_rel_abs = np.asarray(err_rel_abs, dtype=np.float32)
#er_dict = {'evento': n_event_test['evento'], 'etiqueta_pos': test, 'pred_pos': np.around(pred, 2), 'residuo': er}
#er_df   = pd.DataFrame(data = er_dict)
#er_df.to_csv(results_fold + nn_type + '/error_' + time.replace(' ', '_') + sub_set_train + '_' + sub_set_test + '.csv') 

print('el error absoluto es: {}'.format(err_abs))
#print('el error absoluto (km) es: {}'.format(110.574*err_abs)) # convert latitude into km Latitude: 1 deg = 110.574 km
print('el error relativo es: {}'.format(err_rel))

#vect_dist = dist_vect(y_test, pred)
#dist_km = [np.around(degrees2kilometers(vect_dist[j][2]),2) for j in range(pred.shape[0])]

er_dict = {'names_events': n_event_test[:,1], 'station': [id_2station(st) for st in station_test[:,0]], 
           'etiqueta_m': test, 
           'pred_m': pred[:], 'er_rel': err_rel_abs*100}
print(test.shape, pred[:].shape)
er_df   = pd.DataFrame(data = er_dict)
er_df.to_csv('resultados/m_' + time.replace(' ', '_') + '_prediccion_pendiente_evento_estacion.csv')

###################################################


'''
best_epoch = np.argmin(loss_val)

epochs = np.linspace(1, len(loss_train), num = len(loss_train))
fig0, ax0 = plt.subplots( nrows=1, ncols=1, tight_layout = True)
ax0.plot(epochs, loss_train, 'g', label = 'loss train')
ax0.plot(epochs, loss_val, 'b', label = 'loss val')
ax0.set_title(f"learning rate = %f \n best epoch = %f" % (lr, best_epoch + 1))
ax0.set_xlabel('epoch')
ax0.set_ylabel('score')
plt.legend()
plt.show()
fig0.savefig(results_fold + nn_type + '/rendimiento_' + time.replace(' ', '_') + sub_set_train + '_' + sub_set_test + '.pdf')

##################################################

train_val = sub_set_train
test      = sub_set_test
epoch     = len(epochs)

results = {"time":time,"pc":pc,"nn_type":nn_type,"train_val":train_val,"test":test, 'epochs': epoch, "err_abs":err_abs,"err_rel":err_rel}

tablename = results_fold + "historico_{lr}.csv".format(lr=lr)
if path.exists(tablename) == False:
    df = pd.DataFrame(columns=results.keys())
    df.to_csv(tablename,index = False)
df = pd.read_csv(tablename)
df_results = pd.DataFrame(data=results,index=[0])
df = pd.concat([df,df_results], ignore_index=False)
df.to_csv(tablename,index = False)
'''
'''
er=110.574*np.asarray(er)
fig, ax = plt.subplots( nrows=1, ncols=3, tight_layout = True)
ax[0].scatter(x= y_test[:,1], y=y_test[:,0], c= 'b', label='datos')
ax[0].scatter(x=pred[:,1], y=pred[:,0], c='g', marker = 'o', label='modelo generado')
ax[0].set_xlabel('longitude')
ax[0].set_ylabel('latitude')
ax[0].set_ylim([-56, -17])
ax[0].set_xlim([-75, -66])
ax[0].grid(True)
ax[0].legend()
ax[1].scatter(x=er[:,1], y=er[:,0], c ='k', label='residuos')
ax[1].set_xlabel('residuo longitude')
ax[1].set_ylabel('residuo latitude')
#ax[1].set_xlim([-17, -56])
ax[1].legend()
ax[1].grid(True)
ax[2].text(1,1, 'el error absoluto (grados) es: {}'.format(np.around(err_abs, 2)))
ax[2].text(1,11, 'el error absoluto (km) es: {}'.format(np.around(110.574*err_abs, 3)))
ax[2].text(1,21, 'el error relativo es: {}'.format(err_rel))
ax[2].set_xlim([0,12])
ax[2].set_ylim([0,25])
plt.show()
fig.savefig('resultados/dist_' + time.replace(' ', '_') + '_prediccion_localizacion_evento_estacion.pdf')

lat_0 = np.mean(y_test[:,0])
lon_0 = np.mean(y_test[:,1])

zoom_scale = 5

bbox = [lat_0 - 2*zoom_scale, lat_0 + 2*zoom_scale,\
        lon_0 - zoom_scale, lon_0 + zoom_scale]

#print('este es el primer valor de vect_dist: ', vect_dist[0])
fig = plt.figure(figsize=(8, 16))

m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\
            llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')

#m   = Basemap(projection='lcc', resolution='h', 
#            lat_0=lat_0, lon_0=lon_0,
#            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(linewidth=2.0, color='gray')
m.drawcoastlines(color='gray') # draw coastlines
m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)

#m.drawmapboundary(color='gray') # draw a line around the map region

m.scatter(y_test[:,1], y_test[:,0], latlon=True,
          s=50,
          cmap='Greens', alpha=0.8,
          label='datos')

# pred
m.scatter(pred[:,1], pred[:,0], latlon=True,
          s=50,
          cmap='Reds', alpha=0.8,
          label='modelo')

for i in range(pred.shape[0]):
    vect_dist_i = vect_dist[i]
    m.plot(vect_dist_i[1], vect_dist_i[0], latlon = True, linestyle = 'dashed', color = 'r', linewidth = 1.0, alpha = 0.5)

plt.legend()
plt.show()
#print([np.around(degrees2kilometers(vect_dist[j][2]),2) for j in range(pred.shape[0])])
fig.savefig('resultados/dist_' + time.replace(' ', '_') + '_prediccion_localizacion_evento_estacion.pdf')
'''