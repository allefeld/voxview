import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


# T1-weighted, values 0 to 4095
vol = nib.load("/home/ca/Store/lab/dicom/nifti/07 T1w_MPR GR_IR.nii")
t1w = vol.get_fdata().flatten()
# T2-weighted, values 0 to 2745
vol = nib.load("/home/ca/Store/lab/dicom/nifti/09 T2w_SPC SE.nii")
t2w = vol.get_fdata().flatten()

plt.hist2d(t1w, t2w, bins=300, norm=mpl.colors.LogNorm())
plt.colorbar()


kmeans = KMeans(n_clusters=4)
x = np.vstack((t1w, t2w)).transpose()
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(t1w, t2w, s=1, c=y_kmeans)
