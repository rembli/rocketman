import os
import pandas as pd
import yaml

def main ():
    data_root = "C:\Data\Dev-Data\music\\"
    if os.getenv("ROCKETMAN_DATA") is not None:
        data_root = os.getenv("ROCKETMAN_DATA")
    write_labels(data_root+"labels\\")

def write_labels (path):
    filenames = os.listdir(path)

    csv = []
    for filename in filenames:
        if filename != "excluded" and os.path.isdir(path + filename):
            filenames2 = os.listdir(path + filename)
            for filename2 in filenames2:
                row = [filename, path + filename, filename2]
                csv.append(row)

    df = pd.DataFrame(data=csv, columns=['label', 'path', 'filename'])
    df.to_csv(path + "labels.csv", index=None, header=True)
    print(df)

if __name__ == '__main__':
    main()


