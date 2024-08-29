import os
import pickle
from PIL import Image
import random


def isValidPath(root_dir):
    flag = False
    files = os.listdir(root_dir)
    if len(files) == 12 * 2:
        try:
            for i in range(12):
                id = str(i).zfill(3)
                image_path = os.path.join(root_dir, id + '.png')
                image = Image.open(image_path).convert("RGB")
            flag = True
        except:
            return flag
    return flag


if __name__ == "__main__":
    # set data path, views_release is directly downloaded from the Zero123.
    root_dir = '/data/zero123/views_release/'
    neg_save_file = '/data/zero123/negative_pairs.pkl'
    obj_path_file = '/data/zero123/obj_path.pkl'
    num_pairs = 1e8 # the maximum number of negative pairs

    # Extract valid obj path
    obj_path = list()
    for model_name in os.listdir(root_dir):
        if isValidPath(os.path.join(root_dir, model_name)):
            obj_path.append(model_name)
    obj_path.sort()
    with open(obj_path_file, 'wb') as f:
        pickle.dump(obj_path, f)

    # filepath list and index table
    # for each obj: 000.npy/png - 011.npy/png
    obj_path = pickle.load(open(obj_path_file, 'rb'))
    assert obj_path == obj_path, print("different obj path order")
    obj_index = list(range(len(obj_path)))

    n_pairs = set()
    count = 0
    while count < num_pairs:
        ref = random.choice(obj_index)
        cur = random.choice(obj_index)
        if ref == cur:
            continue
        pair = (ref, cur) if ref < cur else (cur, ref)
        n_pairs.add(pair)
        count += 1
        if count % 100000 == 0:
            print("Finish {} pairs ... target {} pairs".format(count, num_pairs))

    with open(neg_save_file, 'wb') as f:
        pickle.dump(list(n_pairs), f)
    print("Done (generating negative pairs)")

