import json
import yaml
import shutil
import h5py

from fcutils._file import raise_on_path_not_exists, pathify, return_list_smart

# ----------------------------------- find ----------------------------------- #
@pathify
@raise_on_path_not_exists
@return_list_smart
def subdirs(folderpath, pattern="*"):
    """
        returns all sub folders in a given folder matching a pattern
    """
    return [f for f in folderpath.glob(pattern) if f.is_dir()]


@pathify
@raise_on_path_not_exists
@return_list_smart
def files(folderpath, pattern="*"):
    """
        returns all files folders in a given folder matching a pattern
    """
    return [f for f in folderpath.glob(pattern) if f.is_file()]


# --------------------------------- deleting --------------------------------- #
@pathify
@raise_on_path_not_exists
def delete(path):
    """
        Deletes either a folder or a file
    """
    if path.is_dir():
        shutil.rmtree(str(path))
    else:
        path.unlink()


# ---------------------------------- saving ---------------------------------- #
@pathify
def to_json(filepath, obj):
    """ saves an object to json """
    with open(filepath, "w") as out:
        json.dump(obj, out)


@pathify
def to_yaml(filepath, obj):
    """ saves an object to yaml """
    with open(filepath, "w") as out:
        yaml.dump(obj, out, default_flow_style=False, indent=4)


# ---------------------------------- loading --------------------------------- #
@pathify
@raise_on_path_not_exists
def from_json(filepath):
    """ loads an object from json """
    with open(filepath, "r") as fin:
        return json.load(fin)


@pathify
@raise_on_path_not_exists
def from_yaml(filepath):
    """ loads an object from yaml """
    with open(filepath, "r") as fin:
        return yaml.load(fin, Loader=yaml.FullLoader)


@pathify
@raise_on_path_not_exists
def open_hdf(filepath):
    """
        Open and expand from a hdf file
    """
    f = h5py.File(filepath, "r")

    # List all groups and subgroups
    keys = list(f.keys())
    subkeys = {}
    for k in keys:
        try:
            subk = list(f[k].keys())
            subkeys[k] = subk
        except:
            pass

    all_keys = []
    f.visit(all_keys.append)

    return f, keys, subkeys, all_keys
