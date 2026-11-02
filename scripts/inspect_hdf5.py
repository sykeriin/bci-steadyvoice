"""Print the structure of the first session found, so the T15 loader's assumptions can be checked."""
import glob, os, sys
import h5py
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True))
print(len(files), "hdf5 files; first 5:", files[:5])
with h5py.File(files[0], "r") as f:
    k = sorted(f.keys())[0]
    print("first trial key:", k, "| n trials:", len(f.keys()))
    for name, ds in f[k].items(): print("  dataset", name, ds.shape, ds.dtype)
    print("  attrs:", dict(f[k].attrs))
