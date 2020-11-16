from glob import glob

def what_we_have():
    return [company.split("\\")[1].split("_")[0] for company in glob("static/cnn_models/*")]