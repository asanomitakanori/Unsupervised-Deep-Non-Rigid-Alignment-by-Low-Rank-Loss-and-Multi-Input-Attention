import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
from math import ceil
import imageio


def convert_3ch(src):
    '''Return 3ch image
    Args:
        src (numpy.ndarray): Input image
    '''
    chk = src.shape
    return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if len(chk) == 2 or chk[-1] == 1 else src


def weighted_sum(srcs, rates=None, save_name=None):
    '''Sum the images
    Args:
        srcs (list of numpy.ndarray): Image list to be synthesized
        rates (list of float): Percentage list to be synthesized (rate1+rate2 can be more than 1)
    '''
    if len(srcs):
        raise Exception(f'The size of srcs is zero. srcs: {len(srcs)}, rates: {len(rates)}')
    if rates is None:
        rates = [1.0 / len(srcs)] * len(srcs)
    if len(srcs) != len(rates):
        raise Exception(f'The lengths of srcs and rates are different. srcs: {len(srcs)}, rates: {len(rates)}')

    src_list = [convert_3ch(src) for src in srcs]
    out = 0
    for src, rate in zip(src_list, rates):
        out = out + src * rate
    out[out > 255] = 255
    out = out.astype('uint8')
    if save_name is not None:
        cv2.imwrite(save_name, out)
    else:
        return out


def get_img_table(srcs, clm=3, save_name=None, thk=3, lcolor=(255, 255, 255), scolor=(0, 0, 0)):
    '''Arrange the given image list in the specified number of columns.
    Args:
        srcs (list of numpy.ndarray): Image list
        clm (int): Number of columns
        save_name (str): Save the input with its name, return it if None.
        thk (int): Thickness of table line
        lcolor (tuple): Color of table line
        scolor (tuple): Color of surplus cell
    Return:
        itable (numpy.ndarray): Image table if save_name is None
    Example:
        >>> itable = get_img_table([im1,im2,im3], clm=2, thk=1, lcolor=(0,0,255), scolor=(255,0,0))
    '''

    vbar = np.full((srcs[0].shape[0], thk, 3), lcolor, np.uint8) if thk else None
    hbar = np.full((thk, (srcs[0].shape[1] + thk) * clm - thk, 3), lcolor, np.uint8) if thk else None

    surplus = np.full(convert_3ch(srcs[0]).shape, scolor, np.uint8)
    if thk:
        surplus = np.hstack([surplus, vbar])
    itable = []

    for i in range(len(srcs)):
        srcs[i] = convert_3ch(srcs[i])
        srcs[i] = np.hstack([srcs[i], vbar]) if thk else srcs[i]
    for i in range(clm - len(srcs) % clm):
        srcs.append(surplus)

    for l in range(ceil(len(srcs) / clm)):
        c_imgs = np.hstack(srcs[l * clm:l * clm + clm])
        if thk:
            c_imgs = c_imgs[:, :-thk]
        itable.append(c_imgs)
        itable.append(hbar)
    itable = np.vstack(itable)
    if thk:
        itable = itable[:-thk]
    if save_name is not None:
        cv2.imwrite(save_name, itable)
        return
    else:
        return itable


def get_video(srcs, save_name, fps=10):
    black_bar = np.zeros((35, srcs[0].shape[1], 3)).astype('uint8')
    black_bar[-2:] = 255
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (srcs[0].shape[1], srcs[0].shape[0])
    video = cv2.VideoWriter(save_name, fourcc, fps, size)
    for i, src in enumerate(srcs):
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if (len(src.shape) == 2) or (src.shape[-1] == 1) else src
        video.write(src)
    video.release()


def get_gif(srcs, save_name, optim=False, duration=40, loop=0):
    images = []
    for src in srcs:
        chk = src.shape
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if len(chk) == 2 or chk[-1] == 1 else src
        images.append(Image.fromarray(src[..., [2, 1, 0]]))
    # images[0].save(save_name, save_all=True, append_images=images[1:], optimize=optim, duration=duration, loop=loop)
    imageio.mimsave(save_name, images, 'GIF', **{'duration': duration})


def put_text(src, text, point, font_path='/home/asanomi/デスクトップ/functions/e_mage/sample_fonts/KosugiMaru-Regular.ttf', font_size=12, color=(255, 255, 255),
             anchor='lt'):
    '''Write text on an images
    Args:
        src (numpy.ndarray): Image
        text (str): Text
        point (tuple of interger): Text coordinates, default anchor is the upper left of the text box.
        font_path (str): Font path，e.g.: https://fonts.google.com/
        font_size (int): Font size
        color (tuple): Text color
        anchor (str): Refer to https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors
    '''

    font = ImageFont.truetype(font_path, font_size)

    chk = src.shape
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) if len(chk) == 2 or chk[-1] == 1 else src

    src = np.uint8(src)
    # Convert to pil_image
    pil_image = Image.fromarray(src[..., [2, 1, 0]])

    draw = ImageDraw.Draw(pil_image)
    draw.text(point, text, fill=color, font=font, anchor=anchor)

    cv_result_image = np.asarray(pil_image)[..., [2, 1, 0]]

    return cv_result_image

