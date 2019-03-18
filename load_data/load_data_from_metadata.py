import os
import json

def main():
    arg = sys.argv[1]
    if arg not in ["leukemia", "colon"]:
        raise Exception("undefined type")

    type = arg

    raw_dir = "{}_raw_data".format(type)
    metadata_file = "metadata/metadata.{}.json".format(type)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    ids = []
    for m in metadata:
        ids.append(m['file_id'])

    file_size = len(ids)//10

    for i in range(10):
        if i < 9:
            output = {"ids":ids[file_size*i:file_size*(i+1)]}
        else:
            output = {"ids":ids[file_size*i:]}

        filepath = os.path.join("request", "request{}.txt".format(i) )
        with open(filepath, 'w') as outfile:
            json.dump(output, outfile)

    command = "curl --remote-name --remote-header-name"
    file_endpt =  "https://api.gdc.cancer.gov/data/"

    request_dir = "./request"
    request_file_list = os.listdir(request_dir)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    # change directory to save raw data
    os.chdir(raw_dir)

    for file in request_file_list:
        if ".txt" not in file:
            continue
        file_path = os.path.join(request_dir, file)
        print(file_path)

        with open(file_path, "r") as f:
            data = json.load(f)
            uuid_list = data['ids']
            uuid = ",".join(uuid_list)
        
            os.system("{} '{}{}'".format(command, file_endpt, uuid))


if __name__ == "__main__":
    main()
