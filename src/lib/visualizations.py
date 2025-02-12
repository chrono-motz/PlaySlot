"""
Visualization functions
"""

import os
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib import colors
import imageio
from PIL import Image
from torchvision.utils import draw_segmentation_masks
from webcolors import name_to_rgb
from IPython.display import display, Image as IPImage
import io

import lib.utils as utils
from CONFIG import COLORS, COLOR_MAP



def visualize_sequence(sequence, savepath=None, tag="sequence", add_title=True, add_axis=False, n_cols=10,
                       size=3, font_size=11, n_channels=3, titles=None, tb_writer=None, iter=0,
                       vmax=1, vmin=0, **kwargs):
    """ Visualizing a grid with several images/frames """

    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)

    figsize = kwargs.pop("figsize", (size*n_cols, size*n_rows))
    fig.set_size_inches(*figsize)
    if("suptitle" in kwargs):
        suptitle_size = kwargs.pop("suptitle_size", 12)
        fig.suptitle(kwargs["suptitle"], fontsize=suptitle_size)
        del kwargs["suptitle"]

    ims = []
    fs = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        if n_rows > 1 and n_cols == 1:
            a = ax[row]
        elif n_cols > 1 and n_rows == 1:
            a = ax[col]
        elif n_cols == 1 and n_rows == 1:
            a = ax
        else:
            a = ax[row, col]
        f = sequence[i].permute(1, 2, 0).cpu().detach()
        if(n_channels == 1):
            f = f[..., 0]
        else:
            f = f.clamp(0, 1)
        im = a.imshow(f, vmin=vmin, vmax=vmax, **kwargs)
        ims.append(im)
        fs.append(f)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title, fontsize=font_size)
            else:
                a.set_title(f"Frame {i}", fontsize=font_size)

    # removing axis
    if(not add_axis):
        for row in range(n_rows):
            for col in range(n_cols):
                if n_rows > 1 and n_cols == 1:
                    a = ax[row]
                elif n_cols > 1 and n_rows == 1:
                    a = ax[col]
                elif n_cols == 1 and n_rows == 1:
                    a = ax
                else:
                    a = ax[row, col]
                if n_cols * row + col >= n_frames:
                    a.axis("off")
                else:
                    a.set_yticks([])
                    a.set_xticks([])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        img_grid = torch.stack(fs).permute(0, 3, 1, 2)
        tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
    return fig, ax, ims



def visualize_recons(imgs, recons, savepath=None,  tag="recons", n_cols=10, tb_writer=None,
                     iter=0, fontsize=12):
    """ Visualizing original imgs, recons and error """
    B, C, H, W = imgs.shape
    imgs = imgs.cpu().detach()
    recons = recons.cpu().detach()
    n_cols = min(B, n_cols)

    fig, ax = plt.subplots(nrows=3, ncols=n_cols)
    fig.set_size_inches(w=n_cols * 3, h=3 * 3)
    for i in range(n_cols):
        a = ax[:, i] if n_cols > 1 else ax
        a[0].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
        a[1].imshow(recons[i].permute(1, 2, 0).clamp(0, 1))
        err = (imgs[i] - recons[i]).sum(dim=-3)
        a[2].imshow(err, cmap="coolwarm", vmin=-1, vmax=1)
        a[0].set_xticks([])
        a[0].set_yticks([])
        a[1].set_xticks([])
        a[1].set_yticks([])
        a[2].set_xticks([])
        a[2].set_yticks([])

    ax[0, 0].set_ylabel("Targets", fontsize=fontsize)
    ax[1, 0].set_ylabel("Preds", fontsize=fontsize)
    ax[2, 0].set_ylabel("Difference", fontsize=fontsize)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        tb_writer.add_images(fig_name=f"{tag}_imgs", img_grid=np.array(imgs), step=iter)
        tb_writer.add_images(fig_name=f"{tag}_recons", img_grid=np.array(recons), step=iter)
    return fig, ax



def visualize_aligned_slots(recons_objs, savepath=None, fontsize=16, mult=3):
    """
    Visualizing the reconstructed objects after alignment of slots.

    Args:
    -----
    recons_objs: torch Tensor
        Reconstructed objects (objs * masks) for a sequence after alignment.
        Shape is (num_frames, num_objs, C, H, W)
    """
    T, N, _, _, _ = recons_objs.shape

    fig, ax = plt.subplots(nrows=N, ncols=T)
    fig.set_size_inches((T * mult, N * mult))
    for t_step in range(T):
        for slot_id in range(N):
            ax[slot_id, t_step].imshow(
                    recons_objs[t_step, slot_id].cpu().detach().clamp(0, 1).permute(1, 2, 0),
                    vmin=0,
                    vmax=1
                )
            if t_step == 0:
                ax[slot_id, t_step].set_ylabel(f"Object {slot_id + 1}", fontsize=fontsize)
            if slot_id == N-1:
                ax[slot_id, t_step].set_xlabel(f"Time Step {t_step + 1}", fontsize=fontsize)
            if slot_id == 0:
                ax[slot_id, t_step].set_title(f"Time Step {t_step + 1}", fontsize=fontsize)
            ax[slot_id, t_step].set_xticks([])
            ax[slot_id, t_step].set_yticks([])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax



def visualize_ind_figs(imgs, savepath, tag, size=3):
    """ 
    Visualizing and saving all indiviual frames from a sequence
    """
    N = imgs.shape[0]
    imgs = imgs.cpu().detach()

    for i in range(N):
        plt.figure(figsize=(size, size))
        plt.imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
        plt.axis()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(
                os.path.join(savepath, f"{tag}_{i:02}.png"),
                bbox_inches='tight',
                pad_inches=0.0
            )
        plt.close()
    return



def visualize_ind_figs_objs(objs, masks, savepath, size=3):
    """ 
    Visualizing and saving all indiviual frames from a sequence
    """
    num_frames, num_objs = objs.shape[0], objs.shape[1]
    objs = objs.cpu().detach()
    masks = masks.cpu().detach()

    for frame in range(num_frames):
        for obj in range(num_objs):
            plt.figure(figsize=(size, size))
            plt.imshow(objs[frame, obj].permute(1, 2, 0).clamp(0, 1))
            plt.axis()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                    os.path.join(savepath, f"obj_{obj:02}_{frame:02}.png"),
                    bbox_inches='tight',
                    pad_inches=0.0
                )
            plt.close()
            
            plt.figure(figsize=(size, size))
            plt.imshow(masks[frame, obj].permute(1, 2, 0).clamp(0, 1), cmap="gray")
            plt.axis()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                    os.path.join(savepath, f"mask_{obj:02}_{frame:02}.png"),
                    bbox_inches='tight',
                    pad_inches=0.0
                )
            plt.close()
            
            plt.figure(figsize=(size, size))
            plt.imshow((masks[frame, obj] * objs[frame, obj]).permute(1, 2, 0).clamp(0, 1))
            plt.axis()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                    os.path.join(savepath, f"MaskedObj_{obj:02}_{frame:02}.png"),
                    bbox_inches='tight',
                    pad_inches=0.0
                )
            plt.close()
    return



def visualize_ind_figs_stoch(seed, targets, all_preds, savepath, size=3, is_post=True):
    """ 
    Visualizing and saving all indiviual frames from a sequence
    """
    visualize_ind_figs(seed, savepath=savepath, tag="seed", size=size)
    visualize_ind_figs(targets, savepath=savepath, tag="target", size=size)

    for i in range(len(all_preds)):
        if is_post:
            tag = "post" if i == 0 else f"prior_{i}"
        else:
            tag = f"prior_{i+1}"
        visualize_ind_figs(all_preds[i], savepath=savepath, tag=tag, size=size)
    return



def visualize_decomp(objs, savepath=None, tag="decomp", vmin=0, vmax=1, add_axis=False,
                     size=3, n_cols=10, tb_writer=None, iter=0, ax_labels=None, **kwargs):
    """
    Visualizing object/mask decompositions, having one obj-per-row

    Args:
    -----
    objs: torch Tensor
        decoded decomposed objects or masks. Shape is (B, Num Objs, C, H, W)
    """
    B, N, C, H, W = objs.shape
    n_channels = C
    if B > n_cols:
        objs = objs[:n_cols]
    else:
        n_cols = B
    objs = objs.cpu().detach()

    ims = []
    fs = []
    fig, ax = plt.subplots(nrows=N, ncols=n_cols)
    fig.set_size_inches(w=n_cols * size, h=N * size)
    for col in range(n_cols):
        for row in range(N):
            if N == 1 and n_cols > 1:
                a = ax[col]
            elif N > 1 and n_cols == 1:
                a = ax[row]
            else:
                a = ax[row, col]
            f = objs[col, row].permute(1, 2, 0).clamp(vmin, vmax)
            fim = f.clone()
            if(n_channels == 1):
                fim = fim.repeat(1, 1, 3)
            im = a.imshow(fim, **kwargs)
            ims.append(im)
            fs.append(f)

    for col in range(n_cols):
        if N == 1 and n_cols > 1:
            a = ax[col]
        elif n_cols == 1 and N > 1:
            a = ax[0]
        else:
            a = ax[0, col]
        a.set_title(f"#{col+1}")

    # removing axis
    if(not add_axis):
        for row in range(N):
            for col in range(n_cols):
                if N == 1 and n_cols > 1:
                    a = ax[col]
                elif N > 1 and n_cols == 1:
                    a = ax[row]
                else:
                    a = ax[row, col]
                cmap = kwargs.get("cmap", "")
                a.set_xticks([])
                a.set_yticks([])
            if ax_labels is not None:
                a = ax[row, 0] if n_cols > 1 else ax[row]
                a.set_ylabel(ax_labels[row])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        tb_writer.add_figure(tag=tag, figure=fig, step=iter)
        # img_grid = torch.stack(fs).permute(0, 3, 1, 2)
        # tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
    return fig, ax, ims



def visualize_qualitative_eval(context, targets, preds, savepath=None, context_titles=None, size=4,
                               target_titles=None, pred_titles=None, fontsize=16, n_cols=10):
    """
    Qualitative evaluation of one example. Simultaneuosly visualizing context, ground truth
    and predicted frames.
    """
    n_context = context.shape[0]
    n_targets = targets.shape[0]
    n_preds = preds.shape[0]

    n_cols = min(n_cols, max(n_targets, n_context))
    n_rows = 1 + ceil(n_preds / n_cols) + ceil(n_targets / n_cols)
    n_rows_pred = 1 + ceil(n_targets / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(w=n_cols*size, h=(n_rows+1)*size)

    context = add_border(x=context, color_name="green", pad=2).permute(0, 2, 3, 1).cpu().detach()
    targets = add_border(x=targets, color_name="green", pad=2).permute(0, 2, 3, 1).cpu().detach()
    preds = add_border(x=preds, color_name="red", pad=2).permute(0, 2, 3, 1).cpu().detach()

    if context_titles is None:
        ax[0, n_cols//2].set_title("Seed Frames", fontsize=fontsize)
    if target_titles is None:
        ax[1, n_cols//2].set_title("Target Frames", fontsize=fontsize)
    if pred_titles is None:
        ax[n_rows_pred, n_cols//2].set_title("Predicted Frames", fontsize=fontsize)

    for i in range(n_context):
        ax[0, i].imshow(context[i].clamp(0, 1))
        if context_titles is not None:
            ax[0, i].set_title(context_titles[i])
    for i in range(n_preds):
        cur_row, cur_col = i // n_cols, i % n_cols
        if i < n_targets:
            ax[1 + cur_row, cur_col].imshow(targets[i].clamp(0, 1))
            if target_titles is not None:
                ax[1 + cur_row, cur_col].set_title(target_titles[i])
        if i < n_preds:
            ax[n_rows_pred + cur_row, cur_col].imshow(preds[i].clamp(0, 1))
            if pred_titles is not None:
                ax[n_rows_pred + cur_row, cur_col].set_title(pred_titles[i])

    for a_row in ax:
        for a_col in a_row:
            a_col.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax



def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    nc, h, w = x.shape[-3:]
    b = x.shape[:-3]

    zeros = torch.zeros if torch.is_tensor(x) else np.zeros
    px = zeros((*b, 3, h+2*pad, w+2*pad))
    color = colors.to_rgb(color_name)
    px[..., 0, :, :] = color[0]
    px[..., 1, :, :] = color[1]
    px[..., 2, :, :] = color[2]
    if nc == 1:
        for c in range(3):
            px[..., c, pad:h+pad, pad:w+pad] = x[:, 0]
    else:
        px[..., pad:h+pad, pad:w+pad] = x
    return px



def make_gif(frames, savepath, n_seed=4, use_border=False):
    """ Making a GIF with the frames """
    with imageio.get_writer(savepath, mode='I', loop=0) as writer:
        for i, frame in enumerate(frames):
            frame = torch.nn.functional.interpolate(frame.unsqueeze(0), scale_factor=2)[0]  # HACK
            up_frame = frame.cpu().detach().clamp(0, 1)
            if use_border:
                color_name = "green" if i < n_seed else "red"
                disp_frame = add_border(up_frame, color_name=color_name, pad=2)
            else:
                disp_frame = up_frame
            disp_frame = (disp_frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            writer.append_data(disp_frame)



def visualize_metric(vals, start_x=0, title=None, xlabel=None, savepath=None, **kwargs):
    """ Function for visualizing the average metric per frame """
    _, ax = plt.subplots(1, 1)
    ax.plot(vals, linewidth=3)
    ax.set_xticks(
            ticks=np.arange(len(vals)),
            labels=np.arange(start=start_x, stop=len(vals) + start_x)
        )
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return



def idx_to_one_hot(x):
    """
    Converting from instance indices into instance-wise one-hot encodings
    """
    num_classes = x.unique().max() + 1
    shape = x.shape
    x = x.flatten().to(torch.int64).view(-1,)
    y = torch.nn.functional.one_hot(x, num_classes=num_classes)
    y = y.view(*shape, num_classes)  # (..., Height, Width, Classes)
    y = y.transpose(-3, -1).transpose(-2, -1)  # (..., Classes, Height, Width)
    return y



def masks_to_rgb(x):
    """ Converting from SAVi masks to RGB images for visualization """
    # we make the assumption that the background is the mask with the most pixels (mode of distr.)
    num_objs = x.unique().max()
    background_val = x.flatten(-2).mode(dim=-1)[0]
    n_labels = num_objs + 1

    if n_labels > len(COLORS):
        colors = COLORS + get_random_colors(n_labels - len(COLORS))
    else:
        colors = COLORS

    imgs = []
    for i in range(x.shape[0]):
        img = torch.zeros(*x.shape[1:], 3)
        for cls in range(num_objs + 1):
            color = colors[cls+1] if cls != background_val[i] else "seashell"
            color_rgb = torch.tensor(name_to_rgb(color)).float()
            img[x[i] == cls, :] = color_rgb / 255
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = imgs.transpose(-3, -1).transpose(-2, -1)
    return imgs



def overlay_segmentations(frames, segmentations, colors, num_classes=None, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if num_classes is None:
        num_classes = segmentations.unique().max() + 1
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs



def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)

    # trying to always make the background of the 'seashell' color
    background_id = segmentation.sum(dim=(-1, -2)).argmax().item()
    cur_colors = colors[1:].copy()
    cur_colors.insert(background_id, "seashell")

    img_with_seg = draw_segmentation_masks(
            img,
            masks=segmentation.to(torch.bool),
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255



def visualize_stoch_frame_figs(context, targets, all_preds, titles=None, savepath=None, **kwargs):
    """
    Qualitative evaluation of different predicted samples from the same sequence.
    Simultaneuosly visualizing context, ground truth, and many predicted frames (best PSNR, best SSIM, ...).
    """
    FONTSIZE = kwargs.get("fontsize", 30)

    n_context = context.shape[0]
    n_preds = targets.shape[0]
    n_pred_seqs = all_preds.shape[0]
    context_title = kwargs.get("context_title", "Seed Frames")
    target_title = kwargs.get("target_title", "Target Frames")
    if titles is not None and all_preds.shape[0] != len(titles):
        raise ValueError(f"Number of pred. sequences {n_pred_seqs} not same as number of titles {len(titles)}")

    # setting up figure
    n_cols = kwargs.get("n_cols", 10)
    n_rows_per_pred = ceil(n_preds / n_cols)
    n_rows = 1 + (n_pred_seqs + 1) * n_rows_per_pred
    fig, ax = plt.subplots(n_rows, n_cols)
    size = kwargs.get("size", 4)
    fig.set_size_inches(w=n_cols*size, h=(n_rows+1)*size)

    # context and ground truth targets have a green frame
    context = add_border(x=context, color_name=COLOR_MAP["context"], pad=2)
    targets = add_border(x=targets, color_name=COLOR_MAP["targets"], pad=2)
    context = context.permute(0, 2, 3, 1).cpu().detach()
    targets = targets.permute(0, 2, 3, 1).cpu().detach()

    # plotting context frames and target frames in the first rows
    ax[0, n_cols//2].set_title(context_title, fontsize=FONTSIZE)
    ax[1, n_cols//2].set_title(target_title, fontsize=FONTSIZE)
    for i in range(n_context):
        ax[0, i].imshow(context[i])
    for i in range(n_preds):
        cur_row, cur_col = i // n_cols, i % n_cols
        ax[1 + cur_row, cur_col].imshow(targets[i])

    # plotting each of the predicted sequuences
    for seq in range(n_pred_seqs):
        cur_offset = 1 + (seq + 1) * n_rows_per_pred
        if titles is not None:
            ax[cur_offset, n_cols//2].set_title(titles[seq], fontsize=FONTSIZE)
        cur_seq = add_border(x=all_preds[seq], color_name=COLOR_MAP["preds"], pad=2)
        cur_seq = cur_seq.permute(0, 2, 3, 1).cpu().detach()
        for i in range(n_preds):
            cur_row, cur_col = i // n_cols, i % n_cols
            ax[cur_offset + cur_row, cur_col].imshow(cur_seq[i])

    # removing axis handles and saving plot
    for a_row in ax:
        for a_col in a_row:
            a_col.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax



def display_projections(points, labels, fig=None, ax=None, legend=None, add_legend=True, marker='o',
                        markersize=15, alpha=1.0):
    """
    Displaying low-dimensional data projections
    """
    if len(np.unique(labels)) > len(COLORS[1:]):
        colors = COLORS[1:] + get_random_colors(len(np.unique(labels)) - len(COLORS[1:]))
    else:
        colors = COLORS[1:] 
    
    legend = [f"Class {l}" for l in np.unique(labels)] if legend is None else legend
    if(ax is None):
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    
    for i, l in enumerate(np.unique(labels)):
        idx = np.where(l==labels)
        ax.scatter(
                points[idx, 0],
                points[idx, 1],
                label=legend[int(i)] if add_legend else None,  # i or l?
                c=colors[l],
                marker=marker,
                s=markersize,
                alpha=alpha
            )
        
    if add_legend:
        ax.legend(loc="best")
    return fig, ax



def get_random_colors(num_colors):
    """ """
    import random
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_colors)]
    return colors



def tensors_to_gif(sequence, filename, duration=100):
    images = []
    for tensor in sequence.numpy():
        # Convert tensor to a PIL Image
        image = Image.fromarray((tensor * 255).transpose(1, 2, 0).astype('uint8'), 'RGB')
        images.append(image)
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
    return



def all_tensors_to_gif(seed_frames, target_frames, all_preds_frames, fpath, duration=200):
    """
    Making GIFs out of the seed, target and predicted frames 
    
    Args:
    -----
    seed_frames: torch tensor
        Seed frames. Shape is (N_seed, C, H, W)
    target_frames: torch tensor
        Target frames. Shape is (N_target, C, H, W)
    all_preds_frames: torch tensor
        Predicted frames using different latent vectors. Shape is (N_different_preds, pred_frames, C, H, W)
    fpath: string
        Path to the directory where the GIF will be stored
    duration: int
        Duration of the GIF in miliseconds
    """
    num_gifs = 1 + len(all_preds_frames)
    num_seed = seed_frames.shape[0]
    num_preds = target_frames.shape[0]
    
    images = []
    
    # seed frames
    for i in range(num_seed):
        cur_seed_frame = add_border(seed_frames[i], color_name="green")
        cur_seed_frame = F.pad(cur_seed_frame, (1, 1, 1, 1))
        cur_seed_frame = cur_seed_frame.repeat(1, 1, num_gifs)

        image = Image.fromarray(
            (cur_seed_frame.cpu().detach().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 'RGB'
        )
        images.append(image)
        
    # predicted frames
    for i in range(num_preds):
        target = F.pad(add_border(target_frames[i], color_name="green"), (1, 1, 1, 1))
        cur_pred_frames = [add_border(all_preds_frames[j, i], color_name="red") for j in range(num_gifs-1)]
        cur_pred_frames = [F.pad(f, (1, 1, 1, 1)) for f in cur_pred_frames]
        cur_pred_frames = torch.cat([target, *cur_pred_frames], dim=-1)
        
        image = Image.fromarray(
            (cur_pred_frames.cpu().detach().numpy() * 255).transpose(1, 2, 0).astype('uint8'), 'RGB'
        )
        images.append(image)

    images[0].save(
        fpath,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    return



def visualize_bins_count(bins, tb_writer=None, iter=0, tag=""):
    """ Displaying a bar plot with the bin count for every action """
    num_actions = len(bins)
    action_range = np.arange(num_actions)
        
    fig, ax = plt.subplots()
    ax.bar(action_range, bins, align='center')
    ax.set_xticks(action_range)
    ax.set_ylabel("Counts")
    ax.set_xlabel("Action ID")
    if tb_writer is not None:
        tb_writer.add_figure(tag=tag, figure=fig, step=iter)        
    return fig, ax



def visualize_distance_between_centroids(centroids):
    """
    Visualizing the distance between centroid 
    """
    num_centroids = centroids.shape[0]
    pairwise_dists = (centroids.unsqueeze(0) - centroids.unsqueeze(1)).pow(2).mean(dim=-1)
    max_dist = pairwise_dists.max()
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(w=num_centroids * 1.25, h=num_centroids * 1.25)
    cax = ax.matshow(pairwise_dists.cpu().detach(), vmin=0, vmax=max_dist * 1.1)
    for (i, j), _ in np.ndenumerate(pairwise_dists.cpu().detach()):
        val_disp = round(pairwise_dists[i, j].item(), 3)
        ax.text(j, i, f"{val_disp}", ha='center', va='center', color='white')
    fig.colorbar(cax)
    return fig, ax



def latent_space_vis(points, labels, protos=None):
    """
    Making a projection figure
    """
    fig, ax = display_projections(
            points=points,
            labels=labels,
            legend=[f"Class {l+1}" for l in labels.unique()],
            markersize=20,
            alpha=0.5
        )
    if protos is not None:
        fig, ax = display_projections(
                points=protos,
                labels=range(len(protos)),
                fig=fig,
                ax=ax,
                legend=[f"Proto {i+1}" for i in range(len(protos))],
                marker='x',
                markersize=50,
                alpha=1.0
            )
    return fig, ax



def process_for_latent_space_vis(codewords, latents, action_idxs):
    """ """
    # removing nans
    codewords, _ = utils.remove_nans(codewords)
    latents, nan_mask = utils.remove_nans(latents)
    action_idxs = action_idxs[~nan_mask]
    num_codewords = len(codewords)
    all_embs = torch.cat([codewords, latents], dim=0)        

    return all_embs, codewords, latents, action_idxs



def display_tensor_gif_in_jupyter(tensor_sequence, duration=100):
    """
    Displays a sequence of PyTorch tensors as an animated GIF in a Jupyter notebook.
    
    Args:
        tensor_sequence (torch.Tensor): A 4D tensor with shape (num_frames, height, width, channels).
        duration (int): Duration for each frame in the GIF in milliseconds.
    """
    # Convert tensor to a list of PIL images
    images = []
    for frame in tensor_sequence:
        # Convert frame to numpy array and ensure it’s in the range [0, 255]
        img = (frame.numpy() * 255).astype('uint8')
        
        # Convert to PIL image
        images.append(Image.fromarray(img))

    # Create a BytesIO object to save the GIF
    gif_buffer = io.BytesIO()
    images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    # Display the GIF
    gif_buffer.seek(0)
    display(IPImage(gif_buffer.read()))



