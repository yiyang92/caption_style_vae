# Split generated test data according to image_id
import os
import argparse
import json


def main(args):
    with open(args.res_file) as rf:
        res = json.load(rf)
    # Annotation file
    with open(args.annot_file) as rf:
        annot = json.load(rf)
    # Get image ids
    annot_imid = set([im_id["id"] for im_id in annot["images"]])
    out_file_list = []
    for caption in res:
        if caption["image_id"] in annot_imid:
            out_file_list.append(caption)
    print("Chosen {} captions".format(len(out_file_list)))
    # save into output file
    resfn = args.res_file.split("/")[-1]
    resfn = resfn.split(".")[0] + "_" + ".json"
    resfn = os.path.join("./results", resfn)
    if os.path.exists(resfn):
        print("")
        os.remove(resfn)
    with open(resfn, 'w') as wj:
        print("saving json file into ", resfn)
        json.dump(out_file_list, wj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("res_file")
    parser.add_argument("annot_file")
    args = parser.parse_args()
    main(args)
