from PIL import Image

def _sample_flow_seq(meta_info, imgsbyvid, frame_nb, sampling_rate):
    vid_name, img_idx = meta_info
    len_vid = len(imgsbyvid[vid_name])

    indices = []
    full_indices = range(0, img_idx)
    if len(full_indices) < frame_nb:
        indices = list(full_indices) + [img_idx for i in range(frame_nb - len(full_indices))]
    else:
        indices = list(full_indices[-frame_nb*sampling_rate::sampling_rate])
        if len(indices) < frame_nb:
            indices += [img_idx for i in range(frame_nb - len(indices))]

    flows = []
    images = []
    for idx in indices:
        flowfile = imgsbyvid[vid_name][0][idx]
        flows.append(Image.open(flowfile).convert('RGB'))

        imgfile = imgsbyvid[vid_name][1][idx]
        images.append(Image.open(imgfile).convert('RGB'))

    # Ordered from further to closer to current image index
    return flows, images
