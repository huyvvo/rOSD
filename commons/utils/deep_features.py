import numpy as np
from scipy.io import loadmat
from math import floor, ceil
from os.path import join
import copy

from PIL import Image

from torch import nn, Tensor
from torchvision import models, transforms
from torchvision.models.vgg import model_urls
model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 
                                                  'http://')


def vgg_preprocessing(batch, to_unit=False):
  """
  Perform the preprocessing on RGB images before passing them 
  into VGG nets to be compatible with inputs in training.

  Parameters:

      batch: Tensor of 3 or 4 dimensions, in range [0..255] or
             [0,1]. Images contained in 'batch' are RGB images and
             of size (H x W x 3).

      to_unit: bool, whether to normalize images in 'batch' into
               the range [0,1]

  Returns:

      A Tensor of the same size with 'batch', ready to be passed
      through VGG nets.
  """

  normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  # copy 'batch' to a new Tensor 
  norm_batch = copy.deepcopy(batch)
  # normalize 'batch' into [0,1] range
  if to_unit:
    norm_batch = norm_batch/255.0

  if len(norm_batch.shape) == 3:
    # convert images to 'torch' format with size (3 x H x W)
    norm_batch = np.transpose(norm_batch, (2,0,1))
    # normalize images with mean and std of ImageNet
    norm_batch = normalizer(norm_batch)
  elif len(norm_batch.shape) == 4:
    # convert images to 'torch' format with size (3 x H x W)
    norm_batch = np.transpose(norm_batch, (0,3,1,2))
    for i in range(norm_batch.shape[0]):
      # normalize images with mean and std of ImageNet
      norm_batch[i] = normalizer(norm_batch[i])
  
  return norm_batch


def roi_rcnn_warping(image, proposals, out_size, p):
  """
  Warp proposals in 'image' into patches of expected size specified
  by 'out_size'. The proposals are warped into patches of size
  out_size + p before being centered cropped into the expected size.

  Parameters:

      image: (H x W x 3) numpy array.

      proposals: (N x 4) numpy array whose each row is a tuple
                 [x1, y1, x2, y2] specifying the position of the
                 corresponding porposal in the image: It is
                 image[y1:y2, x1:x2, :]

      out_size: a pair specifying the expected size of the proposals
                after the warping.

      p: int, cf the description of the function for detail.

  Returns:

    A numpy array containing patches of warped proposals.
  """

  # number of proposals
  n = np.shape(proposals)[0]

  warped_rois = []
  for i in range(n):

    # get info of the current proposal
    x1, y1, x2, y2 = proposals[i]
    w, h = x2-x1, y2-y1

    # get an extension of the proposal before doing the warping
    warp_size = [out_size[0] + 2*p, out_size[1] + 2*p]
    # size of the extension
    ww, hh = [ceil(warp_size[1]/out_size[1] * w), 
              ceil(warp_size[0]/out_size[0] * h)]
    # firstly get an extention of the image to avoid indice exceeds
    # dimension error
    padded_image = np.pad(image, ((floor((hh-h)/2), ceil((hh-h)/2)), 
                           (floor((ww-w)/2), ceil((ww-w)/2)),
                           (0,0)), 
                   'constant')
    # 
    roi_w_context = padded_image[y1:y2+hh-h, x1:x2+ww-w, :]
    warped_roi_w_context = np.array(Image.fromarray(roi_w_context).resize(warp_size))
    warped_roi = warped_roi_w_context[p:-p, p:-p, :]
    assert(tuple(warped_roi.shape[:2]) == tuple(out_size))
    warped_rois.append(warped_roi)

  return np.array(warped_rois)


def rois_to_deep_features(net, image, proposals, 
                          out_size, p, batch_size, verbose=False):
  """
  Transform ROIs into deep features as in R-CNN.

  Parameters:

      net: a deep network with an implemented forward() method 
           which returns the deep feature corresponding a given input.
      
      image: an RGB image of size (H x W x 3) in values in range [0, 255].
      
      proposals: (N x 4) array, each row is [x1,y1,x2,y2] 
                 representing the ROI image[y1:y2, x1:x2, :]

      out_size: a 2-tuple representing the required size of inputs
                to the net.

      p: int, a parameter in warping (cf RCNN for definition).

      batch_size: int, number of proposals for which deep feature
                  is computed in each iteration. batch size should
                  be not too big to avoid running out of memory.

      verbose: bool
      
  Returns:

      A numpy array whos each row is the deep feature of a proposal.
  """

  # get proposals of expected size
  warped_rois = roi_rcnn_warping(image, proposals, out_size, p)
  # pre-processing the proposal before passing into VGG
  warped_rois = vgg_preprocessing(Tensor(warped_rois), True)
  
  # array to contain the final result
  features = []
  # number of proposals
  n = np.shape(proposals)[0]
  # number of iteration needed to compute features for all proposals
  num_batches = ceil(n/batch_size)
  net = net.cuda()

  for _batch in range(num_batches):
    # get current batch
    left_index = _batch * batch_size
    right_index = min((_batch+1)*batch_size, n)
    indices = list(range(left_index, right_index))
    if verbose:
      print('Computing deep features for proposals in range ' + \
            '[%d..%d]'%(indices[0], indices[-1]))
    # pass the batch through the net to obtain deep features
    feat = net.forward(warped_rois[indices].cuda())
    features.append(feat.cpu().detach().numpy())
  return np.concatenate(features, axis=0)


def roi_maxpool_fast_rcnn(featmap, proposals, outsize):
  """
  A simple function performing roi max pooling separately for each
  proposal.

  Parameters:

      featmap: (C x H x W) tensor, a feature map obtained from 
        a deep network for classification after feeding the image
        to it.

      proposals: (N x 4) array representing proposals in the 
        image. Each row is a 4-tuple [x1/H, x2/H, y1/W, y2/W] 
        where (x1,y1) and (x2,y2) are coordinates of upper left
        and lower right corner of the corresponding proposal.

      outsize: 2-tuple or an array of 2 integers (H' x W') indicating
        expecting spatial size of the output. 

  Returns:

    (N x M) array whose rows are fixed size vector representing
    the corresponding proposal. M is equal to C x H' x W'
  """

  N = np.shape(proposals)[0]; # number of proposals
  M = C * outsize[0] * outsize[1]
  rois = np.empty([N, M])
  pooler = AdaptiveMaxPool2d(outsize)
  H, W = np.shape(featmap)[1:3]
  for i in range(N):
    prop = proposals[i]
    x1, x2 = floor(prop[0] * W), ceil(prop[2] * W)
    y1, y2 = floor(prop[1] * H), ceil(prop[3] * H)
    roi = pooler(featmap[:, y1:y2, x1:x2]).view(-1)
    rois[i] = np.array(roi)

  return rois

class FC6(nn.Module):

  def __init__(self):
    vgg = models.vgg16(pretrained=True)
    super(FC6, self).__init__()
    self.features = vgg.features
    self.classifier = nn.Sequential(
                        *list(vgg.classifier.children())[:2]
                      )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

class Pool5(nn.Module):

  def __init__(self):
    vgg = models.vgg16(pretrained=True)
    super(Conv53, self).__init__()
    self.features = vgg.features 

  def forward(self, x, flatten=True):
    x = self.features(x)
    if flatten:
      x = x.view(x.size(0), -1)
      
    return x 

class Pool4(nn.Module):

  def __init__(self):
    vgg = models.vgg16(pretrained=True)
    super(Conv43, self).__init__()
    self.features = nn.Sequential(
                        *list(vgg.features.children())[:24]
                        )

  def forward(self, x, flatten=True):
    x = self.features(x)
    if flatten:
      x = x.view(x.size(0), -1)
    
    return x 

  


if __name__ == '__main__':
  import sys
  sys.path.append('/home/vavo/code/utils')
  import roi_pool
  from imp import reload as rl
  fc6 = deep_features.FC6()
  imgset = 'vocx'
  clname = 'aeroplane_left'
  imdb_path = join('/home/vavo', imgset, clname, 
                   clname + '_small.mat')
  imdb = loadmat(imdb_path)
  img_idx = 2
  image = imdb['images'][img_idx][0]
  bboxes = imdb['bboxes'][img_idx][0]
  bboxes[:,0] -= 1
  bboxes[:,1] -= 1

  rois = roi_pool.roi_rcnn_warping(image, bboxes, [224,224], 16)
  rois = np.transpose(rois, (0,3,1,2))
  feat = fc6.forward(Tensor(rois))
