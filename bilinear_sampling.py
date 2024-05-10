import numpy as np

def bilinear_sampler(source_img, sampling_points):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    # """

    H = source_img.shape[0]
    W = source_img.shape[1]

    x = sampling_points[:,1]
    y = sampling_points[:,0]

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # get pixel value at corner coords
    #In the example where the selected point is (0,0), Ia, Ib, Ic, and Id represent the top left, top right, bottom left, and bottom right pixels of the 2x2 patch
    Ia = source_img[y0, x0]
    Ib = source_img[y1, x0]
    Ic = source_img[y0, x1]
    Id = source_img[y1, x1]

    # recast as float for delta calculation
    x0 = x0.astype(float)
    x1 = x1.astype(float)
    y0 = y0.astype(float)   
    y1 = y1.astype(float)    

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # # add dimension for addition
    wa = wa[...,None]
    wb = wb[...,None]
    wc = wc[...,None]
    wd = wd[...,None]

    # compute output
    stacked = np.stack((wa*Ia, wb*Ib, wc*Ic, wd*Id), axis=0)
    out = np.sum(stacked, axis=0)

    return out



if __name__ =="__main__":
    #Start with some arbitrary image
    image = np.random.rand(256,256,3)
    sampled_points = np.array([[0,0]]) #Top left, top right, bottom left, bottom right
    output = bilinear_sampler(image, sampled_points)