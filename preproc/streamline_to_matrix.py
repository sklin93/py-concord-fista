import sys
import h5py
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.cm as cmx
# import matplotlib.colors as colors


# = = = = = = = = FUNCTION DEFINITION = = = = = = = = = = 
# search the nearest cortex ROI of a certain voxel
def nearest_roi(voxel_coord, atlas_mat, limit):

	x = voxel_coord[0]
	y = voxel_coord[1]
	z = voxel_coord[2]

	rois = atlas_mat[x,y,z]
	if rois > 0:
		# print "voxel roi: (%d)" %(rois)
		return rois, 0

	r = 0
	terminate = 200
	rois = np.array([])
	while rois.size == 0:
		r += 1
		near = atlas_mat[x-r:x+r+1,y-r:y+r+1,z-r:z+r+1]
		rois = near[np.nonzero(near)]
		if r > terminate:
			break

	roi_index = 0
	if r > terminate:
		roi_index = 0
		print "		Bad End-Point!	 "
	else:
		unique, counts = np.unique(rois, return_counts=True)
		roi_index = unique[np.argmax(counts)]
		if r > limit:
			print "		Quality Warning!	"
			print "		nearest roi: (%d), distance: (%d)" %(roi_index, r)

	return roi_index, r


def plot_streamline(ax, sl, roi1, roi2, atlas):

	x,y,z = sl.T

	ax.plot(x,y,z,linewidth=5.0)
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')
	# plt.show()

	voxel = np.where(atlas==roi1)
	x,y,z = voxel

	ax.plot(x,y,z,'ro', alpha=0.1)
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')
	# plt.show()

	voxel = np.where(atlas==roi2)
	x,y,z = voxel

	ax.plot(x,y,z,'go', alpha=0.1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

	return ax



def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color



def plot_atlas(ax, atlas):

	roi_num = atlas.max()
	color_map = get_cmap(roi_num)

	for roi in range(1,roi_num+1):

		voxel = np.where(atlas==roi)
		x,y,z = voxel
		ax.scatter(x,y,z,color=color_map(roi-1),alpha=0.1)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	plt.show()

	return ax




# = = = = = = = = DO WORK = = = = = = = = = = 
# load command argument of input h5 file
# print sys.argv[1:] #-- check input arguments

# # plotting preparation
# from mpl_toolkits.mplot3d import Axes3D
# plt.hold(True)
# ax = plt.subplot(1,1,1,projection='3d')

# load streamline tracking file
f = h5py.File(sys.argv[1],"r")
# print f["streamlines/coords"] #--check if correctly loaded

# load atlas file
f2 = h5py.File(sys.argv[2],'r') 
atlas = f2.get('atlas') 
atlas = np.array(atlas) # For converting to numpy array
# note that the converted atlas matrix is indexed from zero
roi_num = atlas.max()

# # plotting the chosen atlas
# plot_atlas(ax, atlas)

# loop through all streamlines
sl_num = f["streamlines/coords"].shape[0]
print "%d streamlines in total" %(sl_num)
st_mat_a = np.zeros((roi_num,roi_num))
st_mat_q = np.zeros((roi_num,roi_num))
roi_beg = 0
roi_end = 0
invalid_stream = 0
limit = 10

# # streamline examples: 3, 10, 173, 342516,342670
# check_index = [3]
check_index = range(sl_num);

for sl_idx in check_index:
	
	# extract streamline coordinates 
	sl = f["streamlines/coords"][sl_idx].reshape(-1,3)
	
	# query nearest ROI of starting and ending voxels
	roi_beg, r1 = nearest_roi(np.rint(sl[0,:]).astype(int), atlas, limit) 
	roi_end, r2 = nearest_roi(np.rint(sl[-1,:]).astype(int), atlas, limit)

	# adding to structural matrix
	st_mat_a[roi_beg-1, roi_end-1] += 1
	st_mat_a[roi_end-1, roi_beg-1] += 1
	
	# if (roi_beg > 0) and (roi_end > 0):
	if (r1 <= limit) and (r2 <= limit):
		st_mat_q[roi_beg-1, roi_end-1] += 1
		st_mat_q[roi_end-1, roi_beg-1] += 1
	else:
		invalid_stream += 1
		print "bad streamline: No.(%d)" %(sl_idx)

	# # uncomment following if information needs to be printed
	# print "streamline: No.(%d)" %(sl_idx)
	# print "starting from %d, ending to %d" %(roi_beg, roi_end)
	# print "starting dist %d, ending dist %d" %(r1, r2)

	# # uncomment following if streamline plotting is needed
	# ax = plot_streamline(ax, sl, roi_beg, roi_end, atlas)

print "%d bad streamlines in total" %(invalid_stream)

# export structural matrix (all streamlines)
np.savetxt(sys.argv[3]+'.a.csv', st_mat_a)

# export structural matrix (only qualified streamlines)
np.savetxt(sys.argv[3]+'.q.csv', st_mat_q)

# export normalized structural matrix
# elem_num = sl_num - invalid_stream
# st_mat_normalized = np.true_divide(st_mat, elem_num)
# np.savetxt(sys.argv[3], st_mat_normalized)