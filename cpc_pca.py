from sklearn.decomposition import IncrementalPCA
import kaldi_io as ko
import scipy.fftpack as fft
import pickle

''' fit incremental PCA for reducing dimension from 40 to 24 '''

cpc_train = 'cpc/cpc-8/cpc_train.scp' # original cpc feature 
cpc_val   = 'cpc/cpc-8/cpc_validation.scp'
cpc_eval  = 'cpc/cpc-8/cpc_eval.scp'
"""
pca_train = 'cpc/cpc-4/cpc_train'
pca_val   = 'cpc/cpc-4/cpc_validation'
pca_eval  = 'cpc/cpc-4/cpc_eval'
"""
ipca  = IncrementalPCA()

# train pca incrementally with train data
for key,mat in ko.read_mat_scp(cpc_train):
    ipca.partial_fit(mat)

with open('pca/ipca8.pkl', 'wb') as f:
    pickle.dump(ipca, f)
exit()
"""
# fit validation data with pca 
ark_scp_output='ark:| copy-feats ark:- ark,scp:' + pca_val + '.ark,' + pca_val + '.scp'

with ko.open_or_fd(ark_scp_output,'wb') as f:
    for key,mat in ko.read_mat_scp(cpc_val):
        pca_mat = ipca.transform(mat)
        ko.write_mat(f, pca_mat, key=key)

# fit eval data with pca 
ark_scp_output='ark:| copy-feats ark:- ark,scp:' + pca_eval + '.ark,' + pca_eval + '.scp'

with ko.open_or_fd(ark_scp_output,'wb') as f:
    for key,mat in ko.read_mat_scp(cpc_eval):
        pca_mat = ipca.transform(mat)
        ko.write_mat(f, pca_mat, key=key)

# fit train data with pca 
ark_scp_output='ark:| copy-feats ark:- ark,scp:' + pca_train + '.ark,' + pca_train + '.scp'

with ko.open_or_fd(ark_scp_output,'wb') as f:
    for key,mat in ko.read_mat_scp(cpc_train):
        pca_mat = ipca.transform(mat)
        ko.write_mat(f, pca_mat, key=key)
"""
