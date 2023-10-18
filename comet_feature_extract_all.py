import pickle
import numpy as np
from comet.csk_feature_extract import CSKFeatureExtractor

extractor = CSKFeatureExtractor()

for dataset in ['iemocap']:
    print ('Extracting features in CPED')
    # print('Extracting features in', dataset)
    # sentences = pickle.load(open(dataset + '/' + dataset + '_sentences.pkl', 'rb'))
    # feaures = extractor.extract(sentences)
    # pickle.dump(feaures, open(dataset + '/' + dataset + '_features_comet.pkl', 'wb'))
    name = 'cped'
    sentences = pickle.load(open(dataset + '/' + name + '_sentences.pkl', 'rb'))
    feaures = extractor.extract(sentences)
    pickle.dump(feaures, open(dataset + '/' + name + '_features_comet.pkl', 'wb'))
    
print ('Done!')