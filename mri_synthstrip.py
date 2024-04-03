#!/usr/bin/env python

import os
import sys
import argparse
import scipy

from surfa.image.framed import Volume
from surfa.core.array import pad_vector_length
from surfa.transform.geometry import ImageGeometry
from surfa.image.interp import interp_3d_contiguous_linear, interp_3d_fortran_nearest
from surfa.transform.geometry import cast_image_geometry
from surfa.transform.geometry import image_geometry_equal

from types import SimpleNamespace

ref = '''
If you use SynthStrip in your analysis, please cite:
----------------------------------------------------
SynthStrip: Skull-Stripping for Any Brain Image
A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
NeuroImage 206 (2022), 119474
https://doi.org/10.1016/j.neuroimage.2022.119474 

Website: https://w3id.org/synthstrip
'''

# do not wait for third-party imports just to show usage
import torch
import torch.nn as nn
import numpy as np
import surfa as sf

class run():
    def __init__(self, **kwargs):
        defaultKwargs = {'gpu': False, 'no_csf': False, 'border': False, 'model': None }
        args = SimpleNamespace(**{**defaultKwargs, **kwargs})

        # sanity check on the inputs
        if not args.out and not args.mask:
            sf.system.fatal('Must provide at least --out or --mask output flags.')

        # necessary for speed gains (I think)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        # configure GPU device
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            device = torch.device('cuda')
            device_name = 'GPU'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            device = torch.device('cpu')
            device_name = 'CPU'

        # configure model
        print(f'Configuring model on the {device_name}')

        with torch.no_grad():
            model = StripModel()
            model.to(device)
            model.eval()

        # load model weights
        if args.model is not None:
            modelfile = args.model
            print('Using custom model weights')
        else:
            version = '1'
            print(f'Running SynthStrip model version {version}')
            fshome = args.modelPath
            if fshome is None:
                sf.system.fatal('FREESURFER_HOME env variable must be set! Make sure FreeSurfer is properly sourced.')
            if args.no_csf:
                print('Excluding CSF from brain boundary')
                modelfile = os.path.join(fshome, f'synthstrip.nocsf.{version}.pt')
            else:
                modelfile = os.path.join(fshome, f'synthstrip.{version}.pt')
        checkpoint = torch.load(modelfile, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # load input volume
        image = sf.load_volume(args.image)
        print(f'Input image read from: {args.image}')

        # loop over frames (try not to keep too much data in memory)
        print(f'Processing frame (of {image.nframes}):', end=' ', flush=True)
        mask = []
        for f in range(image.nframes):
            print(f + 1, end=' ', flush=True)
            frame = image.new(image.framed_data[..., f])

            ######## FIX ########
            frame = VolumeFixed(data=frame.data, geometry=frame.geom, metadata=frame.metadata)
            # frame = frame.astype(np.dtype('float32'))
            #####################

            # conform image and fit to shape with factors of 64
            conformed = frame.conform(voxsize=1.0, dtype='float32', method='nearest', orientation='LIA').crop_to_bbox()
            target_shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
            conformed = conformed.reshape(target_shape)

            # normalize intensities
            conformed -= conformed.min()
            conformed = (conformed / conformed.percentile(99)).clip(0, 1)

            # predict the surface distance transform
            with torch.no_grad():
                input_tensor = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)
                sdt = model(input_tensor).cpu().numpy().squeeze()

            max_dist = sdt.max().astype(int)
            if args.border >= max_dist:
                print(f'specified border {args.border} greater than max dtrans {max_dist} - computing sdt')
                dif = args.border - (max_dist - 1)
                mask1 = (sdt >= (max_dist - 1))  # region that original sdt has real distances
                dtrans = scipy.ndimage.morphology.distance_transform_edt(mask1) + (max_dist - 2)
                sdt = dtrans  # negative interior distances are lost, but doesn't matter

            ######## FIX ########
            conformed = VolumeFixed(data=conformed.data, geometry=conformed.geom, metadata=conformed.metadata)
            #####################

            # unconform the sdt and extract mask
            sdt = conformed.new(sdt).resample_like(image, fill=100)

            # find largest CC to be safe
            mask.append((sdt < args.border).connected_component_mask(k=1, fill=True))

        # combine frames and end line
        mask = sf.stack(mask)
        print('done')

        # write the masked output
        if args.out:
            image[mask == 0] = np.min([0, image.min()])
            image.save(args.out)
            print(f'Masked image saved to: {args.out}')

        # write the brain mask
        if args.mask:
            image.new(mask).save(args.mask)
            print(f'Binary brain mask saved to: {args.mask}')

        print(ref)

class StripModel(nn.Module):

    def __init__(self,
                 nb_features=16,
                 nb_levels=7,
                 feat_mult=2,
                 max_features=64,
                 nb_conv_per_level=2,
                 max_pool=2,
                 return_mask=False):

        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

def interpolate(source, target_shape, method, affine=None, disp=None, fill=0):
    """
    Interpolate a 3D image given a voxel-to-voxel affine transform and/or a
    dense displacement field.

    Parameters
    ----------
    source : array_like
        4-dimensional source numpy array, with the last dimension representing data frames.
    target_shape : tuple of ints
        Target base shape of interpolated output. Must be a 3D shape.
    method : str
        Interpolation method. Must 'linear' or 'nearest'.
    affine : array_like
        Square affine transform that maps target voxels coordinates to source voxel coordinates.
    disp : array_like
        Dense vector displacement field. Base shape must match target shape.
    fill : scalar
        Fill value for out-of-bounds voxels.

    Returns
    -------
    np.ndarray
        Interpolated image array.
    """
    if affine is None and disp is None:
        raise ValueError('interpolation requires an affine transform and/or displacement field')

    if method not in ('linear', 'nearest'):
        raise ValueError(f'interp method must be linear or nearest, but got {method}')

    if not isinstance(source, np.ndarray):
        raise ValueError(f'source data must a numpy array, but got input of type {source.__class__.__name__}')

    if source.ndim != 4:
        raise ValueError(f'source data must be 4D, but got input of shape {target_shape}')

    target_shape = tuple(target_shape)
    if len(target_shape) != 3:
        raise ValueError(f'interpolated target shape must be 3D, but got {target_shape}')

    # check affine
    use_affine = affine is not None
    if use_affine:
        if not isinstance(affine, np.ndarray):
            raise ValueError(f'affine must a numpy array, but got input of type {source.__class__.__name__}')
        if not np.array_equal(affine.shape, (4, 4)):
            raise ValueError(f'affine must be 4x4, but got input of shape {affine.shape}')
        # only supports float32 affines for now
        affine = affine.astype(np.float32, copy=False)

    # check displacement
    use_disp = disp is not None
    if use_disp:
        if not isinstance(disp, np.ndarray):
            raise ValueError(f'source data must a numpy array, but got input of type {source.__class__.__name__}')
        if not np.array_equal(disp.shape[:-1], target_shape):
            raise ValueError(f'displacement field shape {disp.shape[:-1]} must match target shape {target_shape}')

        # TODO: figure out what would cause this
        if not disp.flags.c_contiguous and not disp.flags.f_contiguous:
            disp = np.asarray(disp, order='F')

        # ensure that the source order is the same as the displacement field
        order = 'F' if disp.flags.f_contiguous else 'C'
        source = np.asarray(source, order=order)

        # make sure the displacement is float32
        disp = np.asarray(disp, dtype=np.float32)

    else:
        # TODO: figure out what would cause this
        if not source.flags.c_contiguous and not source.flags.f_contiguous:
            source = np.asarray(source, order='F')

    # find corresponding function
    order = 'contiguous' if source.flags.c_contiguous else 'fortran'
    interp_func = globals().get(f'interp_3d_{order}_{method}')

    # speeds up if conditionals are computed outside of function (TODO is this even true?)
    shape = np.asarray(target_shape).astype('int64')

    # ensure correct byteorder
    # TODO maybe this should be done at read-time?
    swap_byteorder = sys.byteorder == 'little' and '>' or '<'
    source = source.byteswap().newbyteorder() if source.dtype.byteorder == swap_byteorder else source

    # a few types aren't supported, so let's just convert to float and convert back if necessary
    unsupported_dtype = None
    if source.dtype in (np.bool8,):
        unsupported_dtype = source.dtype
        source = source.astype(np.float32)

    # run the actual interpolation
    # TODO: there's really no need to have a combined affine and deformation function.
    # these should be split up for simplicity sake (might optimize things a bit too)
    source = source.astype(int)
    shape = shape.astype(int)
    resampled = interp_func(source, shape, affine, disp, fill, use_affine, use_disp)

    # if the input type was unsupported but nearest-neighbor interpolation was used,
    # convert back to the original dtype
    if method == 'nearest' and unsupported_dtype is not None:
        resampled = resampled.astype(unsupported_dtype)

    return resampled


class VolumeFixed(Volume):
    def resample_like(self, target, method='linear', copy=True, fill=0):
        """
        Resample to a specified target image geometry.

        Parameters
        ----------
        target : ImageGeometry
            Target image geometry to resample image data into.
        method : {'linear', 'nearest'}
            Image interpolation method.
        copy : bool
            Return copy of image even if target voxel size is already satisfied.
        fill : scalar
            Fill value for out-of-bounds voxels.

        Returns
        -------
        resampled : !class
            Resampled image with updated geometry.
        """
        if self.basedim == 2:
            raise NotImplementedError('resample_like() is not yet implemented for 2D data, contact andrew if you need this')

        # cast to geometries
        source_geom = cast_image_geometry(self)
        target_geom = cast_image_geometry(target)
        if image_geometry_equal(source_geom, target_geom):
            return self.copy() if copy else self

        # compute the voxel-to-voxel affine
        affine = self.geom.world2vox @ target_geom.vox2world

        # this is an optimization to avoid interpolation if it's not needed:
        # commonly, such as when conforming images for preprocessing, images are cropped
        # to fit a given size before inputting them to some model. then, the model spits
        # out some result that must be resampled back into the original image space. however,
        # if image reshaping was the only preprocessing modification (ie. no rotation or resizing),
        # then the result does not need to be interpolated back into the target domain, it just
        # needs to be mapped back to a certain region of the grid. this section checks whether
        # that can be done by first testing if the source and target voxel sizes, rotation, and
        # shear match and if the differences in starting voxel coordinates are near-integers.
        if np.allclose(source_geom.voxsize,  target_geom.voxsize,  atol=1e-5, rtol=0.0) and \
           np.allclose(source_geom.rotation, target_geom.rotation, atol=1e-5, rtol=0.0) and \
           np.allclose(source_geom.shear,    target_geom.shear,    atol=1e-5, rtol=0.0):
            # now check if there is a integer-difference between source and target coordinates
            coord = affine.inv().transform((0, 0, 0))
            coord_rounded = coord.round()
            if np.allclose(coord, coord_rounded, atol=1e-5, rtol=0.0):
                # compute the slicing coordinates defining the matching grid regions
                target_start = coord_rounded.astype(np.int64)
                source_start = np.array([0, 0, 0])
                target_stop = target_start + source_geom.shape
                source_stop = source_start + source_geom.shape

                # refine the slicing coordinate to ensure they don't exceed the target domain
                delta = np.clip(-target_start, 0, None)
                target_start += delta
                source_start += delta
                delta = np.clip(target_stop - target_geom.shape, 0, None)
                target_stop -= delta
                source_stop -= delta

                # convert to actual array slicings
                target_slicing = tuple([slice(a, b) for a, b in zip(target_start, target_stop)])
                source_slicing = tuple([slice(a, b) for a, b in zip(source_start, source_stop)])

                # place data into target shape
                target_data = np.full((*target_geom.shape, self.nframes), fill, dtype=self.dtype)
                target_data[target_slicing] = self.framed_data[source_slicing]
                return self.new(target_data, target_geom)

        # otherwise just do the standard interpolation with the computed affine
        interped = interpolate(source=self.framed_data, target_shape=target_geom.shape,
                               method=method, affine=affine.matrix, fill=fill)
        return self.new(interped, target_geom)
    def resize(self, voxsize, method='linear', copy=True):
        """
        Reslice image to a specified voxel size.

        Parameters
        ----------
        voxsize : scalar or float
            Voxel size in millimeters.
        method : {'linear', 'nearest'}
            Image interpolation method.
        copy : bool
            Return copy of image even if target voxel size is already satisfied.

        Returns
        -------
        resized : !class
            Resized image with updated geometry.
        """
        if self.basedim == 2:
            raise NotImplementedError('resize() is not yet implemented for 2D data, '
                                      'contact andrew if you need this')

        if np.isscalar(voxsize):
            # deal with a scalar voxel size input
            voxsize = np.repeat(voxsize, 3).astype('float')
        else:
            # pad to ensure array has length of 3
            voxsize = np.asarray(voxsize, dtype='float')
            check_array(voxsize, ndim=1, shape=3, name='voxsize')
            voxsize = pad_vector_length(voxsize, 3, 1, copy=False)

        # check if anything needs to be done
        if np.allclose(self.geom.voxsize, voxsize, atol=1e-5, rtol=0):
            return self.copy() if copy else self

        baseshape3D = pad_vector_length(self.baseshape, 3, 1, copy=False)
        target_shape = np.asarray(self.geom.voxsize, dtype='float') * baseshape3D / voxsize
        target_shape = tuple(np.ceil(target_shape).astype(int))

        target_geom = ImageGeometry(
            shape=target_shape,
            voxsize=voxsize,
            rotation=self.geom.rotation,
            center=self.geom.center)
        affine = self.geom.world2vox @ target_geom.vox2world
        ######## FIX ########
        affinematrix = affine.matrix.astype(np.dtype('float32'))
        interped = interpolate(source=self.framed_data, target_shape=target_shape,
                               method=method, affine=affinematrix)
        return self.new(interped, target_geom)
