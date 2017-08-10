# Matching Networks for One Shot Learning 
This repo provides a Pytorch implementation fo the [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) paper.

## Installation of pytorch
The experiments needs installing [Pytorch](http://pytorch.org/)

## Data 
For the Omniglot dataset the download of the dataset is automatic. For the miniImageNet you need to download the ImageNet dataset and execute the script utils.create_miniImagenet.py changing the lines:
```
pathImageNet = '<path_to_downloaded_ImageNet>/ILSVRC2012_img_train'
pathminiImageNet = '<path_to_save_MiniImageNet>/miniImagenet/'
```
And also change the main file option.py line or pass it by command line arguments:
```
parser.add_argument('--dataroot', type=str, default='<path_to_save_MiniImageNet>/miniImagenet/',help='path to dataset')
```


## Installation

    $ pip install -r requirements.txt
    $ python mainOmniglot.py `#Code for OmniGlot`
    $ python mainMiniImageNet.py `#Code for miniImageNet`
    

## Acknowledgements
Special thanks to https://github.com/zergylord and https://github.com/AntreasAntoniou for their Matching Networks implementation. I intend to use some parts for this implementation. More details at https://github.com/zergylord/oneshot and https://github.com/AntreasAntoniou/MatchingNetworks

## Cite
```
@inproceedings{vinyals2016matching,
  title={Matching networks for one shot learning},
  author={Vinyals, Oriol and Blundell, Charles and Lillicrap, Tim and Wierstra, Daan and others},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3630--3638},
  year={2016}
}
```

## Authors

* Albert Berenguel (@aberenguel) [Webpage](https://scholar.google.es/citations?user=HJx2fRsAAAAJ&hl=en)
