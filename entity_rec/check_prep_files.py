import logging
import os
import requests
import time
logging.getLogger().addHandler(logging.StreamHandler())

def init():
    if _no_prep():
        
        zip_destination = os.path.join(os.getcwd(),"prepfiles.zip")
        destination = os.path.join(os.getcwd())

        if os.path.isfile(zip_destination):
            print("not yet the zip file")
            print(os.listdir(os.getcwd()))
            _unzip(zip_destination,destination)
        else:
            _download()

def _no_prep():   
    print("checking if everyting is in place")
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd,"preperation_files")):
        if not os.path.isfile(os.path.join(cwd,"preperation_files","nl-embedding.pckl")):
            print("embedding not yet downloaded")
            return True
        files = os.listdir(os.path.join(cwd,"preperation_files"))
        if len(files) is not 4:
            print("not yet the folder")
            return True
    else:
        return True

    print("all prep files are already in place, cwd:", cwd)
    return False


def _download():
    logging.info("Start downloading preperation files")
    logging.info("This can take a while (~5min)")
    file_id = '1-oSe1f08q2MA6s4ix_5ELFZ6U5wp-aKP'
    zip_destination = os.path.join(os.getcwd(),"prepfiles.zip")
    start_time = time.time()
    download_file_from_google_drive(file_id, zip_destination)
    print(time.time()-start_time)
    destination = os.path.join(os.getcwd())
    # import cloudstorage as gcs 
    # filenames = ['nl-embedding.pckl',
    #             'list_w_cities.txt',
    #             'list_w_first_names.txt',
    #             'list_w_last_names.txt',]
    # os.mkdir('preperation_files')
    # for filename in filenames:
    #     gcs_file = gcs.open("/chatbot-tina.appspot.com/preperation_files/"+filename)
    #     content = gcs_file.read()
    #     with open('preperation_files' + filename) as f:
    #         f.write(content)
    #     gcs_file.close()

    _unzip(zip_destination,destination)

def _unzip(zip_destination,destination):
    import zipfile
    print("unzip start",zip_destination)
    zip_ref = zipfile.ZipFile(zip_destination, 'r')
    zip_ref.extractall(destination)
    zip_ref.close()
    print("unzip done")



def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

