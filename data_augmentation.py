''' Define data augmentation '''


import matplotlib as mpl
mpl.use('Agg')  # plot figures when no screen available

from DigitalPathology.digitalpathology.augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from DigitalPathology.digitalpathology.augmenters.color.contrastaugmenter import ContrastAugmenter
from DigitalPathology.digitalpathology.augmenters.color.hedcoloraugmenter import HedColorAugmenter
from DigitalPathology.digitalpathology.augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from DigitalPathology.digitalpathology.augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from DigitalPathology.digitalpathology.augmenters.passthroughaugmenter import PassThroughAugmenter
from DigitalPathology.digitalpathology.augmenters.spatial.elasticagumenter import ElasticAugmenter
from DigitalPathology.digitalpathology.augmenters.spatial.flipaugmenter import FlipAugmenter
from DigitalPathology.digitalpathology.augmenters.spatial.rotate90augmenter import Rotate90Augmenter
from DigitalPathology.digitalpathology.augmenters.spatial.scalingaugmenter import ScalingAugmenter
from DigitalPathology.digitalpathology.augmenters.color import coloraugmenterbase

import numpy as np
from matplotlib import pyplot as plt
#from scipy.ndimage import imread
#from scipy.misc import imsave
from glob import glob
from os.path import join, basename, dirname, exists
import os
from tqdm import tqdm
import skimage.color
from PIL import Image, ImageEnhance
import shutil
import time

class DataAugmenter(object):

    def __init__(self, augmentation_tag):
        #get list of augs corresponding to the tag
        self.augmenters = define_augmenters(augmentation_tag)
        #get num of augs corresponding to the tag
        self.n_augmenters = len(self.augmenters)

    def augment(self, patch):
        #switch channel to the first dim
        
        patch = patch.transpose((2, 0, 1))
        #loop through a list of augs
        for k in range(self.n_augmenters):
            #select an aug and randomize its sigma
            self.augmenters[k][1].randomize()
            # t = time.time()
            patch = self.augmenters[k][1].transform(patch)
            # print('{t} took {s} logs.'.format(t=self.augmenters[k][0], s=np.log(time.time() - t)), flush=True)  # TODO

        patch = patch.transpose((1, 2, 0))

        return patch

def rgb_to_gray(batch):

    new_batch = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1))
    for i in range(batch.shape[0]):
        new_batch[i, :, :, 0] = skimage.color.rgb2grey(batch[i, ...])

    new_batch = (new_batch * 255.0).astype('uint8')
    return new_batch

def define_augmenters(augmentation_tag):

    if augmentation_tag == 'baseline':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
         ]

    elif augmentation_tag == 'morphology':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hsv':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.1, 0.1), saturation_sigma_range=(-0.2, 0.2), brightness_sigma_range=(-0.2, 0.2))),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hsv-light':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.1, 0.1), saturation_sigma_range=(-0.1, 0.1), brightness_sigma_range=(0, 0))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hsv-medium':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.3, 0.3), saturation_sigma_range=(-0.3, 0.3), brightness_sigma_range=(0, 0))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hsv-strong':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-1, 1), saturation_sigma_range=(-1, 1), brightness_sigma_range=(0, 0))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    # elif augmentation_tag == 'hsv2':
    #
    #      augmenters = [
    #         ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
    #         ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
    #         ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
    #         ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
    #         ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-1, 1), saturation_sigma_range=(-1, 1), brightness_sigma_range=(-1, 1))),
    #         ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
    #         ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
    #      ]

    elif augmentation_tag == 'hsv_only':

         augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.2, 0.2), saturation_sigma_range=(-0.2, 0.2), brightness_sigma_range=(0, 0))),
         ]

    # TODO change name to hsv_all
    elif augmentation_tag == 'hsv_strong':

         augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-1, 1), saturation_sigma_range=(-1, 1), brightness_sigma_range=(-1, 1))),
            # ('brightness', BrightnessAugmenter(sigma_range=[0.65, 1.35])),
            # ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5]))
         ]

    elif augmentation_tag == 'hsv_strong_extended':

         augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-1, 1), saturation_sigma_range=(-1, 1), brightness_sigma_range=(-1, 1))),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hsv_strong_extended2':

         augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-1, 1), saturation_sigma_range=(-1, 1), brightness_sigma_range=(-1, 1))),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1)))
         ]

    elif augmentation_tag == 'hsv_strong_extended3':

        augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.3, 0.3), saturation_sigma_range=(-0.3, 0.3), brightness_sigma_range=(0, 0))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
        ]

    elif augmentation_tag == 'hed-light-bc-only':

         augmenters = [
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.05, 0.05), haematoxylin_bias_range=(-0.05, 0.05),
                                            eosin_sigma_range=(-0.05, 0.05), eosin_bias_range=(-0.05, 0.05),
                                            dab_sigma_range=(-0.05, 0.05), dab_bias_range=(-0.05, 0.05),
                                            cutoff_range=(0.15, 0.85))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5]))
         ]

    elif augmentation_tag == 'hsv-light-bc-only':

         augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.1, 0.1), saturation_sigma_range=(-0.1, 0.1), brightness_sigma_range=(0, 0))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5]))
         ]

    elif augmentation_tag == 'hed-light':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.05, 0.05), haematoxylin_bias_range=(-0.05, 0.05),
                                            eosin_sigma_range=(-0.05, 0.05), eosin_bias_range=(-0.05, 0.05),
                                            dab_sigma_range=(-0.05, 0.05), dab_bias_range=(-0.05, 0.05),
                                            cutoff_range=(0.15, 0.85))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hed-medium':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.1, 0.1), haematoxylin_bias_range=(-0.1, 0.1),
                                            eosin_sigma_range=(-0.1, 0.1), eosin_bias_range=(-0.1, 0.1),
                                            dab_sigma_range=(-0.1, 0.1), dab_bias_range=(-0.1, 0.1),
                                            cutoff_range=(0.15, 0.85))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hed-strong':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.2, 0.2), haematoxylin_bias_range=(-0.2, 0.2),
                                            eosin_sigma_range=(-0.2, 0.2), eosin_bias_range=(-0.2, 0.2),
                                            dab_sigma_range=(-0.2, 0.2), dab_bias_range=(-0.2, 0.2),
                                            cutoff_range=(0.15, 0.85))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hed':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.05, 0.05), haematoxylin_bias_range=(-0.05, 0.05),
                                            eosin_sigma_range=(-0.05, 0.05), eosin_bias_range=(-0.05, 0.05),
                                            dab_sigma_range=(-0.05, 0.05), dab_bias_range=(-0.05, 0.05),
                                            cutoff_range=(0.15, 0.85))),
            ('hsb_color_b', HsbColorAugmenter(hue_sigma_range=(0, 0), saturation_sigma_range=(0, 0), brightness_sigma_range=(-0.2, 0.2))),
            ('contrast', ContrastAugmenter(sigma_range=(-0.25, 0.25))),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    elif augmentation_tag == 'hed_only':

         augmenters = [
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.05, 0.05), haematoxylin_bias_range=(-0.05, 0.05),
                                            eosin_sigma_range=(-0.05, 0.05), eosin_bias_range=(-0.05, 0.05),
                                            dab_sigma_range=(-0.05, 0.05), dab_bias_range=(-0.05, 0.05),
                                            cutoff_range=(0.15, 0.85))),
            # ('hsb_color_b', HsbColorAugmenter(hue_sigma_range=(0, 0), saturation_sigma_range=(0, 0), brightness_sigma_range=(-0.2, 0.2))),
            # ('contrast', ContrastAugmenter(sigma_range=(-0.25, 0.25)))
         ]

    elif augmentation_tag == 'bc':

        augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50,
                                         interpolation_order=1)),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1))),

        ]

    elif augmentation_tag == 'bc_only':

        augmenters = [
            # ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            # ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            # ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            # ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50,
            #                              interpolation_order=1)),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            # ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            # ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1))),

        ]

    elif augmentation_tag == 'bc-light':

        augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50,
                                         interpolation_order=1)),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1))),

        ]

    elif augmentation_tag == 'bc-medium':

        augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50,
                                         interpolation_order=1)),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.35, 1.65])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.25, 1.75])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1))),

        ]

    elif augmentation_tag == 'bc-strong':

        augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50,
                                         interpolation_order=1)),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0, 2])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0, 2])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1))),

        ]

    elif augmentation_tag == 'all-light':

         augmenters = [
            ('rotate90', Rotate90Augmenter(k_list=[0, 1, 2, 3])),
            ('flip', FlipAugmenter(flip_list=['none', 'vertical', 'horizontal', 'both'])),
            ('scaling', ScalingAugmenter(scaling_range=(0.8, 1.2), interpolation_order=1)),
            ('elastic', ElasticAugmenter(sigma_interval=(9.0, 11.0), alpha_interval=(80.0, 120.0), map_count=50, interpolation_order=1)),
            ('hed_stain', HedColorAugmenter(haematoxylin_sigma_range=(-0.05, 0.05), haematoxylin_bias_range=(-0.05, 0.05),
                                            eosin_sigma_range=(-0.05, 0.05), eosin_bias_range=(-0.05, 0.05),
                                            dab_sigma_range=(-0.05, 0.05), dab_bias_range=(-0.05, 0.05),
                                            cutoff_range=(0.15, 0.85))),
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(-0.1, 0.1), saturation_sigma_range=(-0.1, 0.1), brightness_sigma_range=(0, 0))),
            ('brightness_enh', BrightnessEnhAugmenter(sigma_range=[0.65, 1.35])),
            ('contrast_enh', ContrastEnhAugmenter(sigma_range=[0.5, 1.5])),
            ('additive', AdditiveGaussianNoiseAugmenter(sigma_range=(0.0, 0.1))),
            ('blur', GaussianBlurAugmenter(sigma_range=(0.1, 1)))
         ]

    # TODO remove
    elif augmentation_tag[:10] == 'hsv_manual':

        augs = augmentation_tag.split('#')
        hue = float(augs[1])
        sat = float(augs[2])
        bri = float(augs[3])

        augmenters = [
            ('hsb_color', HsbColorAugmenter(hue_sigma_range=(hue, hue), saturation_sigma_range=(sat, sat), brightness_sigma_range=(bri, bri))),
        ]

    elif augmentation_tag == 'none':

        augmenters = [
            ('none', PassThroughAugmenter())
        ]

    else:
        raise Exception('Unknown augmentation tag: {tag}'.format(tag=augmentation_tag))

    return augmenters

#----------------------------------------------------------------------------------------------------

def plot_augmentation(x_path, y_path, output_dir, n_patches, augmentation_tag, color_space, batch_size=10):

    from utils.data_generator import SupervisedSequence

    gen = SupervisedSequence(
        x_path=x_path,
        y_path=y_path,
        batch_size=batch_size,
        augmenter=DataAugmenter(augmentation_tag),
        one_hot=True,
        compare_augmentation=True,
        color_space=color_space
    )

    if not exists(output_dir):
        os.makedirs(output_dir)

    counter = 0
    for x, y in gen:
        x = x * 0.5 + 0.5
        for i, patch in enumerate(x):
            plt.imshow(patch if patch.shape[-1] == 3 else patch[:, :, 0])
            plt.title(y[i])
            plt.imsave(join(output_dir, '%d.png' % counter), patch if patch.shape[-1] == 3 else patch[:, :, 0], vmin=0, vmax=1, cmap='gray' if color_space == 'grayscale' else None)
            counter += 1

        if counter > n_patches:
            break

#----------------------------------------------------------------------------------------------------

class BrightnessEnhAugmenter(coloraugmenterbase.ColorAugmenterBase):

    def __init__(self, sigma_range):

        # Initialize base class.
        #
        super().__init__(keyword='brightness')

        # Initialize members.
        #
        self.__sigma_range = None
        self.__sigma = None

        # Save configuration.
        #
        self.__setsigmaranges(sigma_range=sigma_range)

    def __setsigmaranges(self, sigma_range):

        # Store the setting.
        #
        self.__sigma_range = sigma_range

    def transform(self, patch):

        # Prepare
        rgb_space = True if patch.shape[0] == 3 else False
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        image = Image.fromarray(np.uint8(patch_image if rgb_space else patch_image[:, :, 0]))

        # Change brightness
        enhanced_image = ImageEnhance.Brightness(image)
        enhanced_image = enhanced_image.enhance(self.__sigma)

        # Convert back
        patch_enhanced = np.asarray(enhanced_image)
        patch_enhanced = patch_enhanced if rgb_space else patch_enhanced[:, :, np.newaxis]
        patch_enhanced = patch_enhanced.astype(dtype=np.uint8)
        patch_enhanced = np.transpose(a=patch_enhanced, axes=(2, 0, 1))

        return patch_enhanced

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)


#----------------------------------------------------------------------------------------------------

class ContrastEnhAugmenter(coloraugmenterbase.ColorAugmenterBase):

    def __init__(self, sigma_range):

        # Initialize base class.
        #
        super().__init__(keyword='contrast_enh')

        # Initialize members.
        #
        self.__sigma_range = None
        self.__sigma = None

        # Save configuration.
        #
        self.__setsigmaranges(sigma_range=sigma_range)

    def __setsigmaranges(self, sigma_range):

        # Store the setting.
        #
        self.__sigma_range = sigma_range

    def transform(self, patch):

        # Prepare
        rgb_space = True if patch.shape[0] == 3 else False
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        image = Image.fromarray(np.uint8(patch_image if rgb_space else patch_image[:, :, 0]))

        # Change brightness
        enhanced_image = ImageEnhance.Contrast(image)
        enhanced_image = enhanced_image.enhance(self.__sigma)

        # Convert back
        patch_enhanced = np.asarray(enhanced_image)
        patch_enhanced = patch_enhanced if rgb_space else patch_enhanced[:, :, np.newaxis]
        patch_enhanced = patch_enhanced.astype(dtype=np.uint8)
        patch_enhanced = np.transpose(a=patch_enhanced, axes=(2, 0, 1))

        return patch_enhanced

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)


#----------------------------------------------------------------------------------------------------

class ColorEnhAugmenter(coloraugmenterbase.ColorAugmenterBase):

    def __init__(self, sigma_range):

        # Initialize base class.
        #
        super().__init__(keyword='color_enh')

        # Initialize members.
        #
        self.__sigma_range = None
        self.__sigma = None

        # Save configuration.
        #
        self.__setsigmaranges(sigma_range=sigma_range)

    def __setsigmaranges(self, sigma_range):

        # Store the setting.
        #
        self.__sigma_range = sigma_range

    def transform(self, patch):

        # Prepare
        rgb_space = True if patch.shape[0] == 3 else False
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        image = Image.fromarray(np.uint8(patch_image if rgb_space else patch_image[:, :, 0]))

        # Change brightness
        enhanced_image = ImageEnhance.Color(image)
        enhanced_image = enhanced_image.enhance(self.__sigma)

        # Convert back
        patch_enhanced = np.asarray(enhanced_image)
        patch_enhanced = patch_enhanced if rgb_space else patch_enhanced[:, :, np.newaxis]
        patch_enhanced = patch_enhanced.astype(dtype=np.uint8)
        patch_enhanced = np.transpose(a=patch_enhanced, axes=(2, 0, 1))

        return patch_enhanced

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigma = np.random.uniform(low=self.__sigma_range[0], high=self.__sigma_range[1], size=None)


#----------------------------------------------------------------------------------------------------

def demonstrate_augmentations(data_dir, output_dir, organ_tags, augmentation_tags, n_batches, batch_size, color_space, rep):

    from utils.data_generator import SupervisedSequence
    import gc

    # For each dataset
    for organ_tag in organ_tags:

        # Data
        x_path = join(data_dir, organ_tag, 'patches', 'validation_x.npy')
        y_path = join(data_dir, organ_tag, 'patches', 'validation_y.npy')

        # Clear memory
        data_seq = None
        gc.collect()

        # For each augmentation
        for augmentation_tag in augmentation_tags:

            # Read sequence
            if data_seq is None:
                data_seq = SupervisedSequence(
                    x_path=x_path,
                    y_path=y_path,
                    batch_size=batch_size,
                    augmenter=DataAugmenter(augmentation_tag),
                    one_hot=True,
                    compare_augmentation=False,
                    color_space=color_space
                )
            else:
                data_seq.augmenter = DataAugmenter(augmentation_tag)  # avoid reading the npy array again

            # Output dir
            result_dir = join(output_dir, organ_tag, augmentation_tag)
            if not exists(result_dir):
                os.makedirs(result_dir)

            # Get different augmentations for same patch
            for k in range(rep):

                # Augment patches
                for i in range(n_batches):

                    # Get data
                    x, _ = data_seq[i]
                    x = x * 0.5 + 0.5

                    # Save
                    for j, patch in enumerate(x):
                        plt.imshow(patch if patch.shape[-1] == 3 else patch[:, :, 0])
                        output_path = join(result_dir, '{b}_{p}_{k}.png'.format(b=i, p=j, k=k))
                        plt.imsave(output_path, patch if patch.shape[-1] == 3 else patch[:, :, 0], vmin=0, vmax=1, cmap='gray' if color_space == 'grayscale' else None)




#----------------------------------------------------------------------------------------------------

#if __name__ == '__main__':

    # augmentations = ['baseline', 'morphology']  # , 'bc-light', 'bc-medium', 'bc-strong', 'hsv-light', 'hsv-medium', 'hsv-strong', 'hed-light', 'hed-medium', 'hed-strong']
    # augmentations = ['hsv_manual#{val}#0#0', 'hsv_manual#0#{val}#0', 'hsv_manual#0#0#{val}']  # , 'bc-light', 'bc-medium', 'bc-strong', 'hsv-light', 'hsv-medium', 'hsv-strong', 'hed-light', 'hed-medium', 'hed-strong']
    # augmentations = ['hsv_only']  # , 'bc-light', 'bc-medium', 'bc-strong', 'hsv-light', 'hsv-medium', 'hsv-strong', 'hed-light', 'hed-medium', 'hed-strong']
    # for augmentation_tag in augmentations:
    #     # for value in [0, 0.1, 0.3, 0.5, 0.8, 1]:

    # augmentation_tag = 'hsv_only'
    # print(augmentation_tag)
    # plot_augmentation(
    #     x_path=r"C:\Users\david\Downloads\data_augmentation_debug\validation_x.npy",
    #     y_path=r"C:\Users\david\Downloads\data_augmentation_debug\validation_y.npy",
    #     output_dir=r'C:\Users\david\Downloads\data_augmentation_debug\speed\{augmentation_tag}'.format(augmentation_tag=augmentation_tag),
    #     # output_dir=r'C:\Users\david\Downloads\data_augmentation_debug\rgb\{augmentation_tag}'.format(augmentation_tag=augmentation_tag.format(val=value)),
    #     n_patches=100,
    #     augmentation_tag=augmentation_tag,
    #     color_space='rgb'
    # )

    # augmentation_tag = 'all-light'
    # root_dir = r'/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/data_augmentation'
    # # root_dir = r'C:\Users\david\Downloads\data_augmentation_debug'
    # print(augmentation_tag)
    # plot_augmentation(
    #     x_path=join(root_dir, "validation_x.npy"),
    #     y_path=join(root_dir, "validation_y.npy"),
    #     output_dir=join(root_dir, '{augmentation_tag}'.format(augmentation_tag=augmentation_tag)),
    #     # output_dir=r'C:\Users\david\Downloads\data_augmentation_debug\rgb\{augmentation_tag}'.format(augmentation_tag=augmentation_tag.format(val=value)),
    #     n_patches=1000,
    #     augmentation_tag=augmentation_tag,
    #     color_space='rgb',
    #     batch_size=64
    # )

    # demonstrate_augmentations(
    #     data_dir=r'/mnt/synology/pathology/projects/DataAugmentationComPat/data',
    #     output_dir=r'/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/data_augmentation_demo/results_rgb',
    #     organ_tags=['mitosis', 'prostate', 'rectum'],
    #     augmentation_tags=["none", "hed-strong", "baseline", "hed-light", "morphology", "hsv-light", "bc", "hsv-strong"],
    #     n_batches=1,
    #     batch_size=32,
    #     color_space='rgb',
    #     rep=10
    # )

    # demonstrate_augmentations(
    #     data_dir=r'/mnt/synology/pathology/projects/DataAugmentationComPat/data',
    #     output_dir=r'/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/data_augmentation_demo/results_grayscale',
    #     organ_tags=['mitosis', 'prostate', 'rectum'],
    #     augmentation_tags=["none", "baseline", "morphology", "bc"],
    #     n_batches=1,
    #     batch_size=32,
    #     color_space='grayscale',
    #     rep=10
    # )
    #
    #demonstrate_augmentations(
    #    data_dir=r'C:\Users\david\Downloads\data_augmentation_debug\data_augmentation_demonstration\data',
    #    output_dir=r'C:\Users\david\Downloads\data_augmentation_debug\data_augmentation_demonstration\network-std',
    #    organ_tags=['lymph'],
    #    augmentation_tags=["none", "hsv_strong"],
    #    n_batches=1,
    #    batch_size=32,
    #    color_space='rgb',
    #    rep=10
    #)
