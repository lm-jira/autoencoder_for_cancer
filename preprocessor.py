import json
import os
import csv
import sys
import pandas
import numpy as np

result_dir = "../data/processed_data"

def generate_csv_from_raw(split = "train"):
    if split not in ["train", "test"]:
        raise Exception("Undefined split")

    # set filename for train/test
    if split == "train":
        data_filename = "training_data.txt"
        metadata_filename = "biospecimen.colon.json"
        rawdata_dir = "./load_data/colon_raw_data"
    else:
        data_filename = "test_data.txt"
        metadata_filename = "biospecimen.leukemia.json"
        rawdata_dir = "./load_data/leukemia_raw_data"

    with open(metadata_filename, "r") as f:
        meta = json.load(f)

    dict = {}
    for m in meta:
        for sample in m["samples"]:
            dict[sample["submitter_id"]] = sample["sample_type"]
    
    with open(os.path.join(result_dir, data_filename), "w") as outfile: 
        for root, subdirList, fileList in os.walk(rawdata_dir):
            for fname in fileList:
                f = fname.split(".")      
       
                #check if file is the methylation data file
                if f[-1] == "txt" and len(f) > 5:

                    submitter = fname.split(".")[5].split("-")[:4]
                    submitter = "-".join(submitter)
                    
                    if submitter in dict.keys():
                        with open( os.path.join(root,fname), "r" ) as raw_file:
                            rawdata = csv.DictReader(raw_file, delimiter="\t")
                            data = []
                            for row in rawdata:
                                data.append(row['Beta_value'])
                                # trya = float(row['Beta_value'])
                            data.append(dict[submitter])

                            if len(data) == 485578: # pick only data with same #features as training data
                                outfile.write("\t".join(data)+"\n")
                        
                    else:
                        #no data in biospecimen (sample_type not defined)
                        print("data not found ", submitter)


def remove_na(split = "train"):
    if split not in ["train", "test"]:
        raise Exception("undefined split")

    csv.field_size_limit(sys.maxsize)
    if split == "train":
        input_filepath = os.path.join(result_dir, "training_data.txt")
        output_filename = "training_data_woNA.txt"
    else:
        input_filepath = os.path.join(result_dir, "test_data.txt")
        output_filename = "test_data_woNA.txt"

    naIndex_path = os.path.join(result_dir, "na_index.txt")

    with open(input_filepath, "r") as f:
        data = csv.reader(f, delimiter="\t")
        
        if split == "train":
            # a set of indices with NA element
            na_indices = set()

            for row in data:
                indices = set([i for i,x in enumerate(row) if x == "NA"])
                na_indices = na_indices.union(indices)  
                
            na_indices = np.array(sorted(list(na_indices)))
            with open(naIndex_path, "w") as out_na:
                out_na.write("\t".join(na_indices.astype(str)))

            f.seek(0)
        else:
            with open(naIndex_path, "r") as in_na:
                na_data = csv.reader(in_na, delimiter="\t")
                row = next(na_data)
                na_indices = np.array(row).astype(int)

#        print("number of all columns = ", len(next(data)))
#        print("number of final columns = ", len(next(data))- len(na_indices))

        outfile_path = os.path.join(result_dir, output_filename)
        with open(outfile_path, "w") as outfile:
            for row in data:
                r = row
                for i in reversed(na_indices):
                    del r[i]

                if split == "test":
                    r = np.array(r)
                    # replace remaining NA with 0
                    na_index = r == "NA"
                    r[na_index] = "0"
                outfile.write("\t".join(r)+"\n")

def test(split = "test"):
    csv.field_size_limit(sys.maxsize)
    if split == "train":
        input_filepath = os.path.join(result_dir, "training_data.txt")
    else:
        input_filepath = os.path.join(result_dir, "test_data.txt")

    label = set()
    with open(input_filepath, "r") as f:
        data = csv.reader(f, delimiter="\t")
        for row in data:
            label = label.union({row[-1]})



def remove_low_variance(split = "train"):
    if split not in ["train", "test"]:
        raise Exception("undefined split")

    csv.field_size_limit(sys.maxsize)
    if split == "train":
        filename = "training_data_woNA.txt"
    else:
        filename = "test_data_woNA.txt"

    file_path = os.path.join(result_dir, filename)
    variance_path = os.path.join(result_dir, "low_variance_index.txt")
    filtered_file_path = os.path.join(result_dir, "filtered_{}".format(filename))

    with open(file_path, "r") as f:
        data = csv.reader(f, delimiter='\t')
    
        if split == "train":
            sum = np.array([])
            num_input = 0.0
            for row in data:
                num_input+=1
                if len(sum)==0:
                    sum = np.append(sum, np.array(row[:-1]).astype('float32') )
                else:
                    sum+= np.array(row[:-1]).astype('float32')
            f.seek(0)
            mean = sum/num_input

            var = np.array([])
            for row in data:
                row_var = (np.array(row[:-1]).astype('float32') - mean )**2

                if len(var)==0:
                    var = np.append(var, row_var)
                else:
                    var+= row_var
            f.seek(0)
            var = var/num_input
            
            var_var = np.var(var)
            threshold = np.mean(var)-2*var_var
            var_indices = var > threshold
            with open(variance_path, "w") as out_var:
                out_var.write("\t".join(var_indices.astype(str)))
        else:
            with open(variance_path, "r") as in_var:
                var_data = csv.reader(in_var, delimiter="\t")
                row = np.array(next(var_data))
                var_indices = row == "True"

        with open(filtered_file_path, "w") as outfile:
            for row in data:
                outfile.write("\t".join(np.array(row[:-1])[var_indices])+"\t"+row[-1]+"\n")



def seperate(split = "train"):

    csv.field_size_limit(sys.maxsize)

    if split == "train":
        data_filepath = os.path.join(result_dir, "filtered_training_data_woNA.txt")
        normal_filepath = os.path.join(result_dir, "filtered_training_normal_data_woNA.txt")
        tumor_filepath = os.path.join(result_dir, "filtered_training_tumor_data_woNA.txt")

        tumor_label = ["Metastatic", "Primary Tumor", "Recurrent Tumor"]
    else:
        data_filepath = os.path.join(result_dir, "filtered_test_data_woNA.txt")
        normal_filepath = os.path.join(result_dir, "filtered_test_normal_data_woNA.txt")
        tumor_filepath = os.path.join(result_dir, "filtered_test_tumor_data_woNA.txt")

        tumor_label = ["Primary Blood Derived Cancer - Peripheral Blood"]

    with open(data_filepath, 'r') as data_file, \
         open(normal_filepath, 'w') as out_normal, \
         open(tumor_filepath, 'w') as out_tumor:
        
        data_list = csv.reader(data_file, delimiter='\t')
        for data in data_list:
            label = data[-1]       

            if label in tumor_label:
                out_tumor.write("\t".join(data[:-1])) # exclude the last element (label)
                out_tumor.write("\n")
            else:
                out_normal.write("\t".join(data[:-1])) # exclude the last element (label)
                out_normal.write("\n")

def pca(split="train"):
    from sklearn.decomposition import PCA
    import pandas as pd

    csv.field_size_limit(sys.maxsize)
    if split == "train":
        input_filepath = os.path.join(result_dir, "training_data.txt")
        output_filename = "training_data_pca.txt"
    else:
        input_filepath = os.path.join(result_dir, "test_data.txt")
        output_filename = "test_data_pca.txt"


    df = pd.read_csv(input_filepath, delimiter="\t")
    label = df[df.columns[-1]]
    df = df[df.columns[:-1]]

    pca = PCA(n_components=100)

    principalComponents = pca.fit_transform(df)
    principalDF = pd.DataFrame(data = principalComponents)
    finalDF = pd.concat([ principalDF, label] , axis=1)

    finalDF.to_csv(output_filename, sep="\t")


def main():
    command = {"gen": "generate_csv_from_raw()", 
               "gen-test": "generate_csv_from_raw('test')",
               "cln": "remove_na('train')",
               "cln-test": "remove_na('test')",
               "sep": "seperate()",
               "sep-test": "seperate('test')",
               "var": "remove_low_variance('train')",
               "var-test": "remove_low_variance('test')",
               "pca": "pca()",
               "pca-test": "pca('test')"}

    arg = sys.argv[1]
    if arg == "test-all":
        generate_csv_from_raw("test")
        print("finish gen")
        remove_na("test")
        print("finish remove")
        remove_low_variance("test")
        print("finish var")
        seperate("test")
    elif arg == "train-all":
        generate_csv_from_raw("train")
        print("finish gen")
        remove_na("train")
        print("finish remove")
        remove_low_variance("train")
        print("finish var")
        seperate("train")    
    elif arg in command.keys():
        print("calling {}".format(command[arg]))
        eval(command[arg])
    else:
#        test()
        print("option not found")

if __name__ == "__main__":
    main()
