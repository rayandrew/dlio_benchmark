import h5py


file = h5py.File("/lus/eagle/projects/MDClimSim/rayandrew/dlio-benchmark/data/stormer/train/img_2919_of_2920.hdf5", "r")
dsname = "records"
chunk = list(file[dsname].shape)
shape = list(file[dsname].shape)
data = file[dsname][:]
file.close()
h5_fname = "/local/scratch/test.hdf5"
h5_file = h5py.File(h5_fname,'w')
dset = h5_file.create_dataset(dsname, shape, data=data, chunks=tuple(chunk))
h5_file.close()


# file = h5py.File("/local/scratch/test.hdf5", "r")
