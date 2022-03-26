import numpy as np
import tensorflow as tf 

def cmvn(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_raw_feature_nps))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = 2**-30
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output

def Escalamiento_minmax(feat_in):
    min_f     = np.min([np.min(x,0) for x in feat_in],0)
    max_f     =  np.max([np.max(x,0) for x in feat_in],0)
    feat_norm = np.array([(feat_in[i]-min_f)/(max_f-min_f) for i in range(len(feat_in))],dtype=object)
    return min_f, max_f, feat_norm

def Escalamiento_standard(feat_in):
    #mean_f = np.mean(feat_in) 
    #std_f  = np.std(feat_in)
    feat_in       = tf.convert_to_tensor(feat_in)
    mean_f, var_f = tf.nn.moments(feat_in, axes = [0, 1, 2, 3])
    std_f         = tf.sqrt(var_f) 
    return float(mean_f), float(std_f)
    

def Escalamiento(feat_in, tipo_de_escalamiento = 'Standard'):

    #'Standard', 'MinMax', 'MVN', 'None'
    # =============================================================================
    # Se Normaliza los features: Para el caso de normalizacion standard se toma el promedio y std
    # Para cada features en la cantidad de frames 
    # =============================================================================
    if tipo_de_escalamiento == 'Standard': #Escalando con Z-score
        print('Se normalizan los features utilizando', tipo_de_escalamiento)
        mean_over_feat = np.mean([np.mean(x) for x in feat_in]) 
        std_over_feat = np.std([np.std(x) for x in feat_in]) 
        feat_norm = np.array([ (feat_in[i]-mean_over_feat)/std_over_feat for i in range(len(feat_in))],dtype=object)
    elif tipo_de_escalamiento == 'MinMax': #Deja entre 0 y 1
        print('Se normalizan los features utilizando', tipo_de_escalamiento)
        min_f     = np.min([np.min(x,0) for x in feat_in],0)
        max_f     =  np.max([np.max(x,0) for x in feat_in],0)
        feat_norm = np.array([(feat_in[i]-min_f)/(max_f-min_f) for i in range(len(feat_in))],dtype=object)
    elif tipo_de_escalamiento == 'MVN':
        feat_norm = np.array([cmvn(feat_in[i]) for i in range(len(feat_in))],dtype=object)
    elif tipo_de_escalamiento == 'None':
        print('No se normalizan los features')
        feat_norm = np.array([feat_in[i] for i in range(len(feat_in))],dtype=object)
    return feat_norm

def unison_shuffled_copies(x, y):
    assert x.shape[0] == len(y)
    p = np.random.permutation(x.shape[0])
    return x[p,:], y[p]

def unison_shuffled_copies_v2(x, y):
    assert x.shape[0] == len(y)
    p = np.random.permutation(x.shape[0])
    x_sh = tf.convert_to_tensor([x[i,:] for i in p], dtype = tf.float32)
    y_sh = tf.convert_to_tensor([y[i] for i in p], dtype = tf.float32)
    return x_sh, y_sh

def unison_shuffled_copies_v3(x, y, z):
    assert x.shape[0] == len(y)
    p = np.random.permutation(x.shape[0])
    x_sh = tf.convert_to_tensor([x[i,:] for i in p], dtype = tf.float32)
    y_sh = tf.convert_to_tensor([y[i] for i in p], dtype = tf.float32)
    z_sh = tf.convert_to_tensor([z[i,:] for i in p], dtype = tf.float32)
    return x_sh, y_sh, z_sh

def unison_shuffled_copies_v4(x, y, z, a):
    assert x.shape[0] == len(y)
    p = np.random.permutation(x.shape[0])
    x_sh = tf.convert_to_tensor([x[i,:] for i in p], dtype = tf.float32)
    y_sh = tf.convert_to_tensor([y[i] for i in p], dtype = tf.float32)
    z_sh = tf.convert_to_tensor([z[i,:] for i in p], dtype = tf.float32)
    a_sh = tf.convert_to_tensor([a[i,:] for i in p], dtype = tf.float32)
    return x_sh, y_sh, z_sh, a_sh      

def zscore_ceps(ceps):
    # shape1 coefs cepstrales
    # shape3 channel
    ceps_norm = []
    for m in range(ceps.shape[0]):
        norm_m = []
        for ch in range(ceps.shape[3]):
            mean_ch = [np.mean(ceps[m, c, :, ch]) for c in range(ceps.shape[1])]
            std_ch  = [np.std(ceps[m, c, :, ch])  for c in range(ceps.shape[1])]
            norm_ch = [(ceps[m, c, :, ch] - mean_ch[c])/std_ch[c] for c in range(ceps.shape[1])]
            norm_m.append(np.array(norm_ch))
        ceps_norm.append(np.array(norm_m))
    ceps_norm = np.array(ceps_norm).astype('float32')
    ceps_norm = ceps_norm.reshape((ceps_norm.shape[0], ceps_norm.shape[2], ceps_norm.shape[3], ceps_norm.shape[1]))
    return ceps_norm

def er_pos(a, b):
    assert len(a)==len(b)
    er = list()
    for i in range(len(a)):
        er_i = np.abs(a[i] - b[i])
        er.append(er_i)
    return er

def station_2id(station):
    id = 0
    if   'AC02'==station: id = 1  
    elif 'CO01'==station: id = 2    
    elif 'CO02'==station: id = 3   
    elif 'GO01'==station: id = 4 
    elif 'GO03'==station: id = 5    
    elif 'MT16'==station: id = 6    
    elif 'PB06'==station: id = 7    
    elif 'PB09'==station: id = 8    
    elif 'PB14'==station: id = 9    
    elif 'PB18'==station: id = 10    
    else: 
        id = 'None'
        print('****ESTACION NO IDENTIFICADA****')
    return id

def id_2station(id):
    station = None
    if   id== 1:  station='AC02'  
    elif id== 2:  station='CO01'    
    elif id== 3:  station='CO02'   
    elif id== 4:  station='GO01' 
    elif id== 5:  station='GO03'    
    elif id== 6:  station='MT16'    
    elif id== 7:  station='PB06'    
    elif id== 8:  station='PB09'    
    elif id== 9:  station='PB14'    
    elif id== 10: station='PB18'     
    else: 
        station = 'None'
        print('****ESTACION NO IDENTIFICADA****')
    return station

def dist_vect(vect_1, vect_2, n = 10):
    dist = []
    for i in range(vect_1.shape[0]):
        lat    = np.linspace(vect_1[i,0],vect_2[i,0], n)
        lon    = np.linspace(vect_1[i,1],vect_2[i,1], n)
        dist_i = np.sqrt((vect_1[i,0] - vect_2[i,0])**2 + (vect_1[i,1] - vect_2[i,1])**2) 
        dist.append([lat,lon, dist_i])
    return dist 

    




            
