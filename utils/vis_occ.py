import os
import numpy as np
from matplotlib.pyplot import imsave

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels,          # semantic occupancy predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    time_step=0,
    save_dir=None,
):
    w, h, z = voxels.shape

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])
    

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
    ]
    #print(len(fov_voxels))

    fov_voxels_2d = fov_voxels[:, [0, 1, 3]]
    image = np.zeros((np.max(fov_voxels_2d[:, 0]).astype(int)+1, np.max(fov_voxels_2d[:, 1]).astype(int)+1))
    for voxel in fov_voxels_2d:
        image[int(voxel[0]), int(voxel[1])] = voxel[2]

    imsave(os.path.join(save_dir, f'vis_{time_step}.png'), image, cmap='jet')
    
    '''
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        # fov_voxels[:, 1],
        # fov_voxels[:, 0],
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=1.0 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=16, # 16
    )

    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    
    mlab.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'))
    mlab.close()'''