"""
read in 256 dimension cpc features --> apply dct --> store 23 dimension feature  

requirement: Kaldi 
"""
import scipy.fftpack as fft
import src.kaldi_io as ko
import argparse

def main():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--input-scp')
    parser.add_argument('--output-scp')
    parser.add_argument('--output-ark')
    parser.add_argument('--dct-dim', type=int)
    args = parser.parse_args()

    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:' + args.output_ark + ',' + args.output_scp
    
    with ko.open_or_fd(ark_scp_output,'wb') as f:
        for key, mat in ko.read_mat_scp(args.input_scp):
            dct_mat = fft.dct(mat, type=2, n=args.dct_dim)
            ko.write_mat(f, dct_mat, key=key)
            
    print('#################success#################')


if __name__ == '__main__':
    main()
